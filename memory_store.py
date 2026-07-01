from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GraphRunRecord:
    run_id: str
    created_at: float
    user_goal: str
    attempts: List[Dict[str, Any]]
    plan: Optional[List[str]] = None
    code: Optional[str] = None
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "user_goal": self.user_goal,
            "attempts": self.attempts,
            "plan": self.plan,
            "code": self.code,
            "stop_reason": self.stop_reason,
        }


class MemoryStore:
    """Hybrid memory.

    - Short-term: in-memory objects passed through graph runner.
    - Long-term: persist run snapshots locally (simulation).

    BigQuery/GCS wiring can be layered later; this keeps the app functional
    without live GCP credentials.
    """

    def __init__(self, base_dir: str = "gcs_bucket_simulation"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.runs_dir = os.path.join(self.base_dir, "agent_graph_runs")
        os.makedirs(self.runs_dir, exist_ok=True)

    def new_run(self, user_goal: str) -> GraphRunRecord:
        run_id = str(uuid.uuid4())
        return GraphRunRecord(
            run_id=run_id,
            created_at=time.time(),
            user_goal=user_goal,
            attempts=[],
        )

    def persist_run(self, record: GraphRunRecord) -> str:
        path = os.path.join(self.runs_dir, f"{record.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)
        return path

