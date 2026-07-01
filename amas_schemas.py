from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    Field = lambda *args, **kwargs: None  # type: ignore
    ValidationError = Exception  # type: ignore


class UserRequestSchema(BaseModel):
    """Validated input for the agent graph."""

    goal: str = Field(..., min_length=3, max_length=4000)
    timeout_seconds: int = Field(5, ge=1, le=60)
    max_retries: int = Field(2, ge=0, le=10)
    token_budget: int = Field(1200, ge=100, le=100000)


class PlannerOutputSchema(BaseModel):
    steps: List[str]


class CodeGenOutputSchema(BaseModel):
    filename: str
    code: str


class ExecutionAttemptSchema(BaseModel):
    attempt_no: int
    ok: bool
    stdout: str = ""
    stderr: str = ""
    traceback: str = ""
    exit_code: Optional[int] = None


class SelfHealDecisionSchema(BaseModel):
    should_retry: bool
    reason: str


class GraphFinalResultSchema(BaseModel):
    ok: bool
    final_output: Optional[str] = None
    attempts: List[ExecutionAttemptSchema]
    plan: Optional[List[str]] = None
    code: Optional[str] = None
    stop_reason: Optional[str] = None


def validate_user_request(data: Dict[str, Any]) -> UserRequestSchema:
    """A tiny wrapper so the app can provide a clean error message."""
    try:
        return UserRequestSchema(**data)
    except ValidationError as e:  # pragma: no cover
        raise ValueError(f"Invalid user request schema: {e}")

