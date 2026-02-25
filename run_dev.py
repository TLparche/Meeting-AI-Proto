from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
HOST = "127.0.0.1"


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _spawn(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    kwargs: dict = {"cwd": str(cwd)}
    if env is not None:
        kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _run_checked(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"명령 실패: {' '.join(cmd)} (exit={completed.returncode})")


def _find_npm_executable() -> str | None:
    return shutil.which("npm.cmd") or shutil.which("npm")


def _is_bindable(host: str, port: int) -> bool:
    if port <= 0 or port > 65535:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, preferred: int, avoid: set[int], scan: int = 100) -> int:
    candidates = [preferred] + list(range(preferred + 1, preferred + scan))
    for port in candidates:
        if port in avoid:
            continue
        if _is_bindable(host, port):
            return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = int(sock.getsockname()[1])
        if port in avoid:
            raise RuntimeError("사용 가능한 포트를 찾지 못했습니다.")
        return port


def _wait_backend_ready(proc: subprocess.Popen, host: str, port: int, timeout_sec: float = 20.0) -> bool:
    deadline = time.time() + timeout_sec
    url = f"http://{host}:{port}/api/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)
    return False


def _start_backend_with_retry(python_exe: str, root: Path, host: str, preferred_port: int) -> tuple[subprocess.Popen, int]:
    tried: set[int] = set()
    for _ in range(8):
        port = _pick_port(host, preferred_port, avoid=tried)
        tried.add(port)
        cmd = [
            python_exe,
            "-m",
            "uvicorn",
            "backend.api:app",
            "--reload",
            "--host",
            host,
            "--port",
            str(port),
        ]
        proc = _spawn(cmd, root)
        if _wait_backend_ready(proc, host, port):
            return proc, port
        _terminate(proc)

    raise RuntimeError("백엔드 포트 바인딩에 실패했습니다. 관리자 권한/보안 소프트웨어/포트 점유 상태를 확인하세요.")


def main() -> int:
    npm_exe = _find_npm_executable()
    if npm_exe is None:
        print("npm을 찾지 못했습니다. Node.js/npm 설치 후 다시 실행하세요.", file=sys.stderr)
        return 1

    if not FRONTEND_DIR.exists():
        print(f"frontend 디렉터리가 없습니다: {FRONTEND_DIR}", file=sys.stderr)
        return 1

    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("[setup] frontend 의존성 설치 중 (npm install)...")
        _run_checked([npm_exe, "install"], FRONTEND_DIR)

    preferred_backend = int(os.environ.get("BACKEND_PORT", "8000"))
    preferred_frontend = int(os.environ.get("FRONTEND_PORT", "5173"))

    try:
        backend_proc, backend_port = _start_backend_with_retry(
            sys.executable,
            ROOT,
            HOST,
            preferred_backend,
        )
    except Exception as exc:
        print(f"[error] backend 시작 실패: {exc}", file=sys.stderr)
        return 1

    frontend_port = _pick_port(HOST, preferred_frontend, avoid={backend_port})
    frontend_cmd = [
        npm_exe,
        "run",
        "dev",
        "--",
        "--host",
        HOST,
        "--port",
        str(frontend_port),
        "--strictPort",
    ]
    frontend_env = os.environ.copy()
    frontend_env["VITE_API_PROXY_TARGET"] = f"http://{HOST}:{backend_port}"

    print(f"[run] backend  : http://{HOST}:{backend_port}")
    print(f"[run] frontend : http://{HOST}:{frontend_port}")
    print("[run] 종료는 Ctrl+C")

    try:
        frontend_proc = _spawn(frontend_cmd, FRONTEND_DIR, env=frontend_env)
    except Exception:
        _terminate(backend_proc)
        raise

    try:
        while True:
            b_code = backend_proc.poll()
            f_code = frontend_proc.poll()
            if b_code is not None or f_code is not None:
                if b_code is None:
                    _terminate(backend_proc)
                if f_code is None:
                    _terminate(frontend_proc)
                return b_code or f_code or 0
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n[stop] 종료 신호 수신. 프로세스를 정리합니다...")
        _terminate(backend_proc)
        _terminate(frontend_proc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
