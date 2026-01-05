"""
LLM Engine - Separate process for running MDM inference
Handles model lifecycle, request queuing, and response delivery
"""

import multiprocessing
import time
import sys
import logging
import logging.handlers

logger = logging.getLogger("LLMEngine")


def setup_child_logging(log_queue: multiprocessing.Queue):
    """Configures logging for child processes to send logs to the main process via queue."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []  # Clear existing handlers (avoid double logging)
    handler = logging.handlers.QueueHandler(log_queue)
    root.addHandler(handler)


def mock_engine_loop(
    request_queue: multiprocessing.Queue,
    response_queue: multiprocessing.Queue,
    log_queue: multiprocessing.Queue,
    ready_event: multiprocessing.Event,
):
    """
    Fast mock loop for testing API and infrastructure without model overhead.
    """
    setup_child_logging(log_queue)
    logger.info("Mock LLM Engine process started")

    # Simulate a short startup delay
    time.sleep(0.5)
    ready_event.set()
    logger.info("Mock Engine ready.")

    while True:
        try:
            result = request_queue.get()
            request_id, prompt = result

            if request_id is None:  # Shutdown signal
                logger.info("Mock Engine received shutdown signal")
                break

            logger.info(f"Processing request {request_id} (MOCK)")

            generated_text = f"Mock response for: '{prompt}'"

            response_queue.put((request_id, generated_text))
            logger.info(f"Finished request {request_id}.")

        except Exception as e:
            logger.error(f"Error in mock engine loop: {e}")


def engine_loop(
    request_queue: multiprocessing.Queue,
    response_queue: multiprocessing.Queue,
    log_queue: multiprocessing.Queue,
    ready_event: multiprocessing.Event,
):
    """
    Main engine loop. Handles input/output queue and performs model loading + generation.
    """
    setup_child_logging(log_queue)
    logger.info("LLM Engine process started")

    try:
        import torch
        from transformers import AutoTokenizer
        from model.modeling_llada import LLaDAModelLM
        from generate import generate_with_dual_cache

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model on {device}...")

        checkpoint = "GSAI-ML/LLaDA-8B-Instruct"

        model = (
            LLaDAModelLM.from_pretrained(
                checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

        logger.info("Model loaded successfully.")
        ready_event.set()

    except Exception as e:
        logger.critical(f"Failed to load model environment: {e}")
        # return 503s permanently on engine bricking
        return

    while True:
        try:
            result = request_queue.get()
            request_id, prompt = result

            if request_id is None:
                logger.info("Engine received shutdown signal")
                break

            logger.info(f"Processing request {request_id}")

            # Sampling parameters (Hardcoded for simplicity)
            gen_length = 128
            steps = 128
            block_length = 32
            temperature = 0.0
            remasking = "low_confidence"

            m = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer(formatted_prompt)["input_ids"]
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # Generate
            with torch.no_grad():
                out, nfe = generate_with_dual_cache(
                    model,
                    input_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    remasking=remasking,
                )

            generated_tokens = out[0][input_ids.shape[1] :]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            response_queue.put((request_id, answer))
            logger.info(f"Finished request {request_id}")

        except Exception as e:
            logger.error(f"Error in engine loop: {e}")
            response_queue.put((request_id, f"Error processing request: {str(e)}"))


class LLMEngine:
    def __init__(self, is_mock: bool = False):
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.log_queue = multiprocessing.Queue()
        self.ready_event = multiprocessing.Event()
        self.process = None
        self.log_listener = None
        self.is_mock = is_mock

    def start(self):
        self.log_listener = logging.handlers.QueueListener(
            self.log_queue, *logging.getLogger().handlers
        )
        self.log_listener.start()

        target_loop = mock_engine_loop if self.is_mock else engine_loop
        loop_name = "LLMEngineWorker_Mock" if self.is_mock else "LLMEngineWorker"

        self.process = multiprocessing.Process(
            target=target_loop,
            args=(
                self.request_queue,
                self.response_queue,
                self.log_queue,
                self.ready_event,
            ),
            daemon=True,
            name=loop_name,
        )
        self.process.start()

    def stop(self):
        if self.process and self.process.is_alive():
            logger.info("Sending shutdown signal to engine...")
            self.request_queue.put((None, None))

            self.process.join(timeout=1.0)

            if self.process.is_alive():
                logger.warning(
                    "Engine did not exit gracefully (likely queue deadlock), forcing termination..."
                )
                self.process.terminate()
                self.process.join()

            logger.info("Engine stopped.")

        if self.log_listener:
            self.log_listener.stop()
            self.log_listener = None

    def submit_request(self, request_id: str, prompt: str):
        self.request_queue.put((request_id, prompt))
