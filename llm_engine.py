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
):
    """
    Fast mock loop for testing API and infrastructure without model overhead.
    """
    setup_child_logging(log_queue)
    logger.info("Mock LLM Engine process started")

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
):
    """
    Main engine loop. Handles input/output queue and performs model loading + generation.
    """
    setup_child_logging(log_queue)
    logger.info("LLM Engine process started")

    while True:
        try:
            result = request_queue.get()
            request_id, prompt = result

            if request_id is None:
                logger.info("Engine received shutdown signal")
                break

            logger.info(f"Processing request {request_id}")

            time.sleep(2.0)
            generated_text = f"Simulated response for: '{prompt}'"

            response_queue.put((request_id, generated_text))
            logger.info(f"Finished request {request_id}.")

        except Exception as e:
            logger.error(f"Error in engine loop: {e}")


class LLMEngine:
    def __init__(self, is_mock: bool = False):
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.log_queue = multiprocessing.Queue()
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
            args=(self.request_queue, self.response_queue, self.log_queue),
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
