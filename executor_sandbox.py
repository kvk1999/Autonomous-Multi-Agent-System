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
    - If Docker is not available, return detectable failure (no host execution).
    """

    if not _docker_available():
        return (
            False,
            2,
            "",
            "",
            "Docker is not available. Sandbox execution was blocked by design.",
        )

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

