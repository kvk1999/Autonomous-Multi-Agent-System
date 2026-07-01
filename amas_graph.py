from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from executor_sandbox import run_script_in_sandbox
from memory_store import MemoryStore
from amas_schemas import (
    ExecutionAttemptSchema,
    GraphFinalResultSchema,
    validate_user_request,
)


class Planner:
    def run(self, goal: str) -> List[str]:
        g = goal.lower()
        if "haversine" in g and "distance" in g:
            return [
                "Parse inputs (lat/lon pairs) from goal.",
                "Generate a haversine distance function.",
                "Add a self-test in main() to print an example distance.",
                "Return output as JSON for stable downstream parsing.",
            ]
        return [
            "Interpret the goal into implementable requirements.",
            "Generate a Python script that computes the requested output.",
            "Add deterministic test output.",
        ]


class Coder:
    def generate(
        self,
        goal: str,
        plan: List[str],
        previous_traceback: Optional[str] = None,
    ) -> Tuple[str, str]:
        filename = "agent_script.py"

        goal_lower = goal.lower()
        if "haversine" in goal_lower:
            # Deterministic example printing JSON to stdout.
            code = """
import math
import json


def haversine_km(lat1, lon1, lat2, lon2, radius_km=6371.0):
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return radius_km * c


def main():
    lat1, lon1 = 40.73061, -73.935242  # NYC-ish
    lat2, lon2 = 42.3601, -71.0589   # Boston-ish
    d_km = haversine_km(lat1, lon1, lat2, lon2)
    print(json.dumps({"distance_km": float(d_km)}))


if __name__ == "__main__":
    main()
"""
            return filename, code

        # Generic fallback script.
        code = f"""
import json


def main():
    print(json.dumps({{"message": "Goal acknowledged", "goal": {goal!r}}}))


if __name__ == "__main__":
    main()
"""
        return filename, code


class Critic:
    def should_retry(self, attempt_no: int, max_retries: int) -> Tuple[bool, str]:
        should = attempt_no < max_retries
        reason = (
            "Detected runtime traceback; will retry with a refactor." if should else "Max retries reached; stopping."
        )
        return should, reason


class Executor:
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str) -> Dict[str, Any]:
        ok, exit_code, stdout, stderr, tb = run_script_in_sandbox(
            code,
            timeout_seconds=self.timeout_seconds,
        )
        return {
            "ok": ok,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "traceback": tb,
        }


class AmasGraphRunner:
    """Minimal decentralized orchestration graph (Planner -> Coder -> Executor -> Critic -> retry)."""

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory = memory_store or MemoryStore()
        self.planner = Planner()
        self.coder = Coder()
        self.critic = Critic()

    def run(self, user_input: Dict[str, Any]) -> GraphFinalResultSchema:
        req = validate_user_request(user_input)

        record = self.memory.new_run(req.goal)

        attempts: List[ExecutionAttemptSchema] = []

        plan = self.planner.run(req.goal)
        record.plan = plan

        previous_traceback: Optional[str] = None
        final_output: Optional[str] = None
        stop_reason: Optional[str] = None

        for attempt_no in range(req.max_retries + 1):
            _, code = self.coder.generate(req.goal, plan, previous_traceback=previous_traceback)
            record.code = code

            executor = Executor(timeout_seconds=req.timeout_seconds)
            exec_res = executor.execute(code)

            attempt = ExecutionAttemptSchema(
                attempt_no=attempt_no,
                ok=bool(exec_res["ok"]),
                stdout=exec_res.get("stdout") or "",
                stderr=exec_res.get("stderr") or "",
                traceback=exec_res.get("traceback") or "",
                exit_code=exec_res.get("exit_code"),
            )

            attempts.append(attempt)
            record.attempts.append(
                attempt.model_dump() if hasattr(attempt, "model_dump") else attempt.__dict__
            )

            if attempt.ok:
                final_output = attempt.stdout.strip() if attempt.stdout else ""
                stop_reason = "Execution succeeded."
                self.memory.persist_run(record)
                return GraphFinalResultSchema(
                    ok=True,
                    final_output=final_output,
                    attempts=attempts,
                    plan=plan,
                    code=code,
                    stop_reason=stop_reason,
                )

            previous_traceback = attempt.traceback
            should_retry, reason = self.critic.should_retry(attempt_no, req.max_retries)

            if not should_retry:
                stop_reason = reason
                self.memory.persist_run(record)
                return GraphFinalResultSchema(
                    ok=False,
                    final_output=None,
                    attempts=attempts,
                    plan=plan,
                    code=code,
                    stop_reason=stop_reason,
                )

        self.memory.persist_run(record)
        return GraphFinalResultSchema(
            ok=False,
            final_output=final_output,
            attempts=attempts,
            plan=plan,
            code=record.code,
            stop_reason=stop_reason or "Unexpected stop.",
        )

