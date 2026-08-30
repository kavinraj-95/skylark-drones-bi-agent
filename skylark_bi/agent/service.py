"""The application service: question in, answer out.

One place that owns the whole pipeline, so the UI stays a thin shell and the same
flow is exercised by tests:

    question -> intent (LLM) -> plan (code) -> metrics (code) -> answer (LLM)

Also owns data loading, including the snapshot fallback. That fallback deserves a
note, because it sits close to something the assignment forbids: the snapshot is
written *only* from a successful live monday.com response, at runtime, to a
gitignored path. It is never committed and never seeds a first run - if monday.com
has never answered, there is nothing to serve and the app says so. Its only job is to
keep a demo alive through a transient outage, always behind an explicit STALE banner.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..analytics.engine import AnalysisResult, execute
from ..analytics.timeframe import PeriodBasis
from ..config import Settings
from ..ingest.builder import build_dataset
from ..ingest.entities import Dataset
from ..monday.client import BoardSnapshot, MondayClient
from ..monday.errors import MondayError
from ..quality.audit import QualityReport, audit
from .llm import LLMClient
from .planner import plan_intent
from .resolver import resolve
from .responder import respond

log = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1


@dataclass
class Answer:
    """Everything the UI needs to render one exchange."""

    question: str
    text: str
    result: AnalysisResult | None = None
    #: Set when the agent needs something from the user before it can answer.
    clarifying_question: str | None = None
    #: Set when the whole attempt failed.
    error: str | None = None
    #: Non-fatal notices worth showing - currently, degradation to keyword/plain mode.
    notes: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.clarifying_question is not None


@dataclass
class LoadedData:
    """A dataset plus its audit and provenance."""

    dataset: Dataset
    quality: QualityReport
    loaded_at: float = field(default_factory=time.monotonic)

    @property
    def is_stale(self) -> bool:
        return self.dataset.is_stale


class DataUnavailableError(RuntimeError):
    """monday.com could not be reached and there is no snapshot to fall back on."""

    def __init__(self, message: str, *, user_message: str) -> None:
        super().__init__(message)
        self.user_message = user_message


class BIService:
    """Owns data loading, caching and the question pipeline."""

    def __init__(self, settings: Settings, *, llm: LLMClient | None = None):
        self._settings = settings
        self._llm = llm
        self._cache: LoadedData | None = None
        #: Answers keyed by (question, data fetch time). The LLM free tier is small and
        #: a demo repeats questions; re-deriving an identical answer would spend quota
        #: for no benefit. Keyed on fetch time so a data refresh invalidates it.
        self._answers: dict[tuple[str, str], Answer] = {}

    # -- data --------------------------------------------------------------------

    @property
    def snapshot_path(self) -> Path:
        return self._settings.cache_dir / "monday.snapshot.json"

    def load(self, *, force_refresh: bool = False) -> LoadedData:
        """Return the current dataset, fetching from monday.com when needed.

        A TTL cache keeps a conversation from becoming N API round-trips - monday's
        daily call budget is finite and a chat would otherwise burn it quickly.
        """
        if (
            self._cache is not None
            and not force_refresh
            and (time.monotonic() - self._cache.loaded_at) < self._settings.data_ttl_seconds
        ):
            return self._cache

        try:
            data = self._fetch_live()
        except MondayError as exc:
            log.warning("Live fetch failed (%s); attempting snapshot fallback.", exc)
            data = self._load_snapshot(reason=exc.user_message)
            if data is None:
                raise DataUnavailableError(
                    f"monday.com unavailable and no snapshot present: {exc}",
                    user_message=(
                        f"{exc.user_message}\n\nThere is also no previously fetched data "
                        "to fall back on, so nothing can be shown yet."
                    ),
                ) from exc

        self._cache = data
        return data

    def _fetch_live(self) -> LoadedData:
        with MondayClient(self._settings.monday) as client:
            deals_id, work_orders_id = client.resolve_board_ids()
            deals = client.get_deals(deals_id)
            work_orders = client.get_work_orders(work_orders_id)

        dataset = build_dataset(deals, work_orders, fetched_at=datetime.now(timezone.utc))
        self._write_snapshot(deals, work_orders)
        return LoadedData(dataset=dataset, quality=audit(dataset))

    def _write_snapshot(self, deals: BoardSnapshot, work_orders: BoardSnapshot) -> None:
        """Persist the raw live response so a later outage has something to serve."""
        try:
            self._settings.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": SNAPSHOT_VERSION,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "boards": {
                    name: {
                        "board_id": snap.board_id,
                        "board_name": snap.board_name,
                        "columns": snap.columns,
                        "items": snap.items,
                    }
                    for name, snap in (("deals", deals), ("work_orders", work_orders))
                },
            }
            self.snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            # A read-only or full filesystem must not break a working live fetch.
            log.warning("Could not write snapshot: %s", exc)

    def _load_snapshot(self, *, reason: str) -> LoadedData | None:
        if not self.snapshot_path.exists():
            return None
        try:
            payload: dict[str, Any] = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            if payload.get("version") != SNAPSHOT_VERSION:
                return None
            boards = payload["boards"]
            snapshots = {
                name: BoardSnapshot(
                    board_id=board["board_id"],
                    board_name=board["board_name"],
                    items=board["items"],
                    columns=board["columns"],
                )
                for name, board in boards.items()
            }
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
        except (OSError, KeyError, ValueError) as exc:
            log.warning("Snapshot unreadable: %s", exc)
            return None

        dataset = build_dataset(
            snapshots["deals"],
            snapshots["work_orders"],
            fetched_at=fetched_at,
            is_stale=True,
            stale_reason=reason,
        )
        return LoadedData(dataset=dataset, quality=audit(dataset))

    # -- questions ---------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Run one question through the full pipeline."""
        question = (question or "").strip()
        if not question:
            return Answer(question=question, text="", error="Please ask a question.")

        try:
            data = self.load()
        except DataUnavailableError as exc:
            return Answer(question=question, text="", error=exc.user_message)

        cache_key = (question.lower(), str(data.dataset.fetched_at))
        cached = self._answers.get(cache_key)
        if cached is not None:
            return cached

        notes: list[str] = []
        intent = plan_intent(question, self._llm, notes)
        plan = resolve(
            intent,
            data.dataset,
            fiscal_start_month=self._settings.fiscal_year_start_month,
            basis=PeriodBasis.FISCAL,
        )

        if plan.needs_clarification:
            # Not cached: a clarification is a prompt to the user, not an answer.
            return Answer(
                question=question,
                text=plan.clarifying_question or "",
                clarifying_question=plan.clarifying_question,
                notes=notes,
            )

        result = execute(plan, data.dataset, data.quality)
        answer = Answer(
            question=question,
            text=respond(question, result, self._llm, notes),
            result=result,
            notes=notes,
        )

        if len(self._answers) > 64:
            self._answers.clear()
        self._answers[cache_key] = answer
        return answer
