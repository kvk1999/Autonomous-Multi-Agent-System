from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from typing import Optional, Tuple


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return True
    except Exception:
        return False


def _run_script_locally_best_effort(
    code: str,
    timeout_seconds: int,
) -> Tuple[bool, int, str, str, str]:
    """Best-effort local execution when Docker is unavailable.

    NOTE: This is not a perfect sandbox; it is used only so the app can
    function in environments where Docker cannot run.
    """

    code = textwrap.dedent(code).strip() + "\n"

    # Clear/limit environment to reduce surprises.
    # - No network blocking is guaranteed, but we reduce likelihood of access.
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""  # avoid accidental GPU usage

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Use isolated mode (-I) to avoid user site packages.
        # Use `py` launcher for Windows compatibility.
        cmd = ["py", "-I", "-u", script_path]


        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            ok = proc.returncode == 0
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            traceback = stderr.strip() if stderr.strip() else ""

            if not ok and not traceback:
                traceback = f"Local execution failed with exit code {proc.returncode}."

            return ok, proc.returncode, stdout, stderr, traceback
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") if hasattr(e, "stdout") else ""
            stderr = (e.stderr or "") if hasattr(e, "stderr") else ""
            return False, 124, stdout, stderr, f"Execution timed out after {timeout_seconds}s (local fallback)."
        except Exception as e:
            return False, 1, "", "", f"Local fallback execution failed: {e}"


def run_script_in_sandbox(
    code: str,
    timeout_seconds: int = 5,
    image: str = "python:3.11-slim",
) -> Tuple[bool, int, str, str, str]:
    """Run untrusted generated Python code.

    Returns:
      ok, exit_code, stdout, stderr, traceback

    Security posture:
    - Prefer Docker ephemeral container.
    - Docker is run with network disabled.
    - If Docker is not available, run locally in a best-effort isolated mode.
    """

    if not _docker_available():
        return _run_script_locally_best_effort(code=code, timeout_seconds=timeout_seconds)


    code = textwrap.dedent(code).strip() + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Mount as read-only.
        # - network disabled by --network none
        # - memory/cpu limits can be added if desired.
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{tmpdir}:/work:ro",
            "-w",
            "/work",
            image,
            "python",
            "-u",
            "script.py",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            ok = proc.returncode == 0

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            # If stderr contains traceback, return it; otherwise keep stderr.
            traceback = stderr.strip() if stderr.strip() else ""
            return ok, proc.returncode, stdout, stderr, traceback
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "") if hasattr(e, "stdout") else ""
            stderr = (e.stderr or "") if hasattr(e, "stderr") else ""
            return False, 124, stdout, stderr, f"Execution timed out after {timeout_seconds}s."
        except Exception as e:
            return False, 1, "", "", f"Sandbox execution failed: {e}" 

