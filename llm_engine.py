"""
LLM Engine - Separate process for running MDM inference
Handles model lifecycle, request queuing, and response delivery
"""

import multiprocessing
import time
import sys
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLMEngine")


def engine_loop(
    request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue
):
    """
    The main loop for the LLM Engine process.
    """
    logger.info("LLM Engine process started.")
    while True:
        try:
            result = request_queue.get()
            request_id, prompt = result

            if request_id is None:  # Shutdown signal
                logger.info("LLM Engine received shutdown signal.")
                break

            logger.info(f"Processing request {request_id}...")

            time.sleep(1.5)
            generated_text = f"Placeholder LLM Generation for prompt: '{prompt}'"

            response_queue.put((request_id, generated_text))
            logger.info(f"Finished request {request_id}.")

        except Exception as e:
            logger.error(f"Error in engine loop: {e}")


class LLMEngine:
    def __init__(self):
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.process = None

    def start(self):
        self.process = multiprocessing.Process(
            target=engine_loop,
            args=(self.request_queue, self.response_queue),
            daemon=True,
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

    def submit_request(self, request_id: str, prompt: str):
        self.request_queue.put((request_id, prompt))
