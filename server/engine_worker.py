import threading
import time
from typing import Dict, Optional

from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig


class EngineWorker:
    def __init__(self, model_path: str, **engine_kwargs):
        engine_kwargs.setdefault("enforce_eager", True)
        self.llm = LLM(model_path, **engine_kwargs)

        self._lock = threading.Lock()
        self._has_work = threading.Event()
        self._stop = threading.Event()

        self.results: Dict[int, dict] = {}
        self.pending: set[int] = set()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, prompt: str | list[int], sp: SamplingParams) -> int:
        # Currently, we use default sparse config for each request
        sparse_config = SparseConfig(
            enabled=True, retention_ratio=0.5, delay_step=1, default_block_len=32
        )
        seq_id = self.llm.add_request(prompt, sp, sparse_configs=sparse_config)
        with self._lock:
            self.pending.add(seq_id)
        self._has_work.set()
        return seq_id

    def status(self, seq_id: int) -> str:
        with self._lock:
            if seq_id in self.results:
                return "finished"
            if seq_id in self.pending:
                return "running"
        return "unknown"

    def get(self, seq_id: int) -> Optional[dict]:
        with self._lock:
            return self.results.get(seq_id)

    def _loop(self):
        while not self._stop.is_set():
            if not self.llm.scheduler.waiting and not self.llm.scheduler.running:
                self._has_work.clear()
                self._has_work.wait(timeout=0.02)
                continue

            # One engine tick
            outputs, _ = self.llm.step()
            if outputs:
                with self._lock:
                    for seq_id, token_ids in outputs:
                        text = self.llm.tokenizer.decode(token_ids)
                        self.results[seq_id] = {
                            "text": text,
                        }
                        self.pending.discard(seq_id)

            # Yield to avoid hogging
            time.sleep(0)

    def shutdown(self):
        self._stop.set()
        self._has_work.set()
        self._thread.join()
        self.llm.exit()
