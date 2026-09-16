"""Shadow deployment ("dark launch") for the MLOps LLM API.

A candidate model runs alongside the active model: every live request is copied
to the shadow candidate in a background worker while the client response always
comes from the active model. Outputs, token usage, latency and agreement with
the active model are recorded to a JSONL log and Prometheus metrics for offline
evaluation, so a new model can be safely validated against production traffic
before it is promoted.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger("mlops_api.shadow")

# ---------------------------------------------------------------------------
# Prometheus metrics for shadow traffic
# ---------------------------------------------------------------------------
SHADOW_REQUESTS = Counter(
    "ml_shadow_requests_total",
    "Shadow deployment request outcomes",
    ["outcome"],
)
SHADOW_LATENCY = Histogram(
    "ml_shadow_duration_seconds",
    "Shadow inference latency",
)
SHADOW_AGREEMENT = Counter(
    "ml_shadow_agreement_total",
    "Shadow prediction agreement with the active model",
    ["agreement"],
)
SHADOW_ERRORS = Counter(
    "ml_shadow_errors_total",
    "Shadow inference errors",
    ["error_type"],
)
SHADOW_QUEUE_LENGTH = Gauge("ml_shadow_queue_length", "Pending shadow jobs in the queue")
SHADOW_MODEL_LOADED = Gauge("ml_shadow_model_loaded", "1 if the shadow model is loaded, 0 otherwise")


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ShadowConfig:
    """Runtime configuration for shadow deployment. Built from env vars by from_env()."""

    enabled: bool = False
    model_name: str = ""
    model_path: str = ""
    log_path: str = "artifacts/shadow/shadow_log.jsonl"
    queue_maxsize: int = 100
    status_entries: int = 50

    @property
    def display_name(self) -> str:
        return self.model_name or self.model_path

    @classmethod
    def from_env(cls) -> "ShadowConfig":
        return cls(
            enabled=_env_flag("SHADOW_ENABLED"),
            model_name=os.getenv("SHADOW_MODEL_NAME", ""),
            model_path=os.getenv("SHADOW_MODEL_PATH", ""),
            log_path=os.getenv("SHADOW_LOG_PATH", "artifacts/shadow/shadow_log.jsonl"),
            queue_maxsize=int(os.getenv("SHADOW_QUEUE_MAX", "100")),
            status_entries=int(os.getenv("SHADOW_STATUS_ENTRIES", "50")),
        )


@dataclass
class ShadowResult:
    model_name: str
    prompt: str
    shadow_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class ShadowJob:
    active_model: str
    prompt: str
    params: Dict[str, Any]
    active_text: str
    active_tokens: int
    request_id: str


@dataclass
class ShadowStats:
    """Process-local counters (mirrored to Prometheus for scraping)."""

    skipped: int = 0
    queued: int = 0
    dropped: int = 0
    completed: int = 0
    failed: int = 0
    identical: int = 0
    differing: int = 0

    def snapshot(self) -> Dict[str, int]:
        return {
            "skipped": self.skipped,
            "queued": self.queued,
            "dropped": self.dropped,
            "processed": self.completed,
            "failed": self.failed,
            "agreement_identical": self.identical,
            "agreement_differing": self.differing,
        }


class ShadowLogStore:
    """Appends shadow comparison records to a JSONL file."""

    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)

    def append(self, entry: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            logger.exception("shadow_log_append_failed path=%s", self.path)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        except OSError:
            logger.exception("shadow_log_read_failed path=%s", self.path)
            return []
        entries: List[Dict[str, Any]] = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries


def _normalize(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def compare_outputs(active_text: Optional[str], shadow_text: Optional[str]) -> str:
    """Bucket active vs shadow output as identical, differing, or error."""
    if active_text is None or shadow_text is None:
        return "error"
    if _normalize(active_text) == _normalize(shadow_text):
        return "identical"
    return "differing"


class ShadowPredictor:
    """Lazily loads and runs a candidate model, fully isolated from the active model."""

    def __init__(self, config: ShadowConfig):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._auto_model_cls = None
        self._auto_tokenizer_cls = None
        SHADOW_MODEL_LOADED.set(0)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _load(self) -> None:
        if self.is_loaded:
            return
        if not self.config.model_name and not self.config.model_path:
            raise RuntimeError(
                "Shadow model is enabled but neither SHADOW_MODEL_NAME nor "
                "SHADOW_MODEL_PATH is set."
            )
        if self._torch is None or self._auto_tokenizer_cls is None or self._auto_model_cls is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM
                from transformers import AutoTokenizer
            except ImportError:
                raise RuntimeError("PyTorch and Transformers are required for shadow inference.")
            self._torch = torch
            self._auto_model_cls = AutoModelForCausalLM
            self._auto_tokenizer_cls = AutoTokenizer

        source = (
            self.config.model_path
            if Path(self.config.model_path).exists()
            else self.config.model_name
        )
        try:
            self._tokenizer = self._auto_tokenizer_cls.from_pretrained(source)
            self._model = self._auto_model_cls.from_pretrained(source)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            SHADOW_MODEL_LOADED.set(1)
            logger.info("shadow_model_loaded source=%s", source)
        except Exception:
            SHADOW_MODEL_LOADED.set(0)
            logger.exception("shadow_model_load_failed source=%s", source)
            raise

    def predict(self, prompt: str, params: Dict[str, Any]) -> ShadowResult:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = tokenizer([text], return_tensors="pt")
        input_length = inputs.input_ids.shape[1]

        start = time.perf_counter()
        no_grad = torch.no_grad() if torch is not None else nullcontext()
        with no_grad:
            outputs = model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens", 150),
                temperature=params.get("temperature", 0.7),
                do_sample=True,
                top_k=params.get("top_k", 50),
                top_p=params.get("top_p", 0.9),
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return ShadowResult(
            model_name=self.config.display_name,
            prompt=prompt,
            shadow_text=generated_text,
            input_tokens=input_length,
            output_tokens=len(generated_tokens),
            latency_ms=latency_ms,
        )


class ShadowDispatcher:
    """Bounded queue + single background worker that runs shadow predictions.

    Live requests enqueue jobs and return immediately; the worker processes them
    serially so shadow inference never adds latency to, or risks, the active path.
    """

    def __init__(self, config: ShadowConfig, predictor: ShadowPredictor, log_store: ShadowLogStore):
        self.config = config
        self.predictor = predictor
        self.log_store = log_store
        self.stats = ShadowStats()
        self._queue = queue.Queue(maxsize=config.queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def queue_size(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if not self.config.enabled or self.running:
            return
        self._thread = threading.Thread(
            target=self._worker,
            name="mlops-shadow-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("shadow_worker_started queue_maxsize=%d", self.config.queue_maxsize)

    def capture(
        self,
        active_model: str,
        prompt: str,
        params: Dict[str, Any],
        active_text: str,
        active_tokens: int,
        request_id: str,
    ) -> bool:
        """Enqueue a shadow job. Never raises; returns True if the job was accepted."""
        if not self.config.enabled:
            with self._lock:
                self.stats.skipped += 1
            SHADOW_REQUESTS.labels(outcome="skipped").inc()
            return False

        job = ShadowJob(
            active_model=active_model,
            prompt=prompt,
            params=params,
            active_text=active_text,
            active_tokens=active_tokens,
            request_id=request_id,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            with self._lock:
                self.stats.dropped += 1
            SHADOW_REQUESTS.labels(outcome="dropped").inc()
            return False
        except Exception:
            logger.exception("shadow_enqueue_failed")
            with self._lock:
                self.stats.dropped += 1
            return False

        with self._lock:
            self.stats.queued += 1
        SHADOW_REQUESTS.labels(outcome="queued").inc()
        SHADOW_QUEUE_LENGTH.set(self._queue.qsize())
        return True

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                start = time.perf_counter()
                result = self.predictor.predict(job.prompt, job.params)
                SHADOW_LATENCY.observe(time.perf_counter() - start)
                self._record_success(job, result)
            except Exception as exc:
                with self._lock:
                    self.stats.failed += 1
                SHADOW_REQUESTS.labels(outcome="shadow_failed").inc()
                SHADOW_ERRORS.labels(error_type=exc.__class__.__name__).inc()
                logger.exception("shadow_prediction_failed request_id=%s", job.request_id)
                self.log_store.append(
                    {
                        "ts": _now_iso(),
                        "request_id": job.request_id,
                        "prompt": job.prompt,
                        "params": job.params,
                        "active": {
                            "model": job.active_model,
                            "text": job.active_text,
                            "tokens": job.active_tokens,
                        },
                        "shadow": {"model": self.config.display_name, "error": str(exc)},
                        "agreement": "error",
                    }
                )
            finally:
                with self._lock:
                    self.stats.completed += 1
                self._queue.task_done()
                SHADOW_QUEUE_LENGTH.set(self._queue.qsize())

    def _record_success(self, job: ShadowJob, result: ShadowResult) -> None:
        agreement = compare_outputs(job.active_text, result.shadow_text)
        with self._lock:
            if agreement == "identical":
                self.stats.identical += 1
            elif agreement == "differing":
                self.stats.differing += 1
        SHADOW_AGREEMENT.labels(agreement=agreement).inc()
        self.log_store.append(
            {
                "ts": _now_iso(),
                "request_id": job.request_id,
                "prompt": job.prompt,
                "params": job.params,
                "active": {
                    "model": job.active_model,
                    "text": job.active_text,
                    "tokens": job.active_tokens,
                },
                "shadow": {
                    "model": result.model_name,
                    "text": result.shadow_text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_ms": round(result.latency_ms, 3),
                },
                "agreement": agreement,
            }
        )

    def flush(self, timeout: float = 10.0) -> None:
        """Drain pending jobs synchronously (used by tests and shutdown)."""
        if not self.config.enabled:
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.005)

    def stop(self) -> None:
        self._stop_event.set()
        if self.running:
            self.flush(timeout=5)
            self._thread.join(timeout=5)
        self._thread = None


class ShadowDeployment:
    """Facade wiring config, predictor, log store and dispatcher together."""

    def __init__(
        self,
        config: Optional[ShadowConfig] = None,
        predictor: Optional[ShadowPredictor] = None,
        log_store: Optional[ShadowLogStore] = None,
        dispatcher: Optional[ShadowDispatcher] = None,
    ):
        self.config = config if config is not None else ShadowConfig.from_env()
        self.predictor = predictor if predictor is not None else ShadowPredictor(self.config)
        self.log_store = log_store if log_store is not None else ShadowLogStore(self.config.log_path)
        self.dispatcher = (
            dispatcher if dispatcher is not None else ShadowDispatcher(self.config, self.predictor, self.log_store)
        )

    def start(self) -> None:
        self.dispatcher.start()

    def stop(self) -> None:
        self.dispatcher.stop()

    def flush(self, timeout: float = 10.0) -> None:
        self.dispatcher.flush(timeout=timeout)

    def capture(
        self,
        prompt: str,
        params: Dict[str, Any],
        active_text: str,
        active_tokens: int,
        request_id: str,
        active_model: str,
    ) -> bool:
        try:
            return self.dispatcher.capture(
                active_model=active_model,
                prompt=prompt,
                params=params,
                active_text=active_text,
                active_tokens=active_tokens,
                request_id=request_id,
            )
        except Exception:
            logger.exception("shadow_capture_failed")
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.config.enabled else "disabled",
            "enabled": self.config.enabled,
            "model_name": self.config.display_name or None,
            "model_ready": self.predictor.is_loaded,
            "queue_length": self.dispatcher.queue_size(),
            "log_path": str(self.config.log_path),
            "recent_entries": self.log_store.recent(limit=self.config.status_entries),
            **self.dispatcher.stats.snapshot(),
        }