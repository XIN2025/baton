import base64
import hashlib
import shutil
import subprocess
from pathlib import Path


def git(ws: Path, *args: str, stdin: bytes | None = None) -> bytes:
    r = subprocess.run(["git", "-C", str(ws), *args], input=stdin, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def prepare(ws: Path, repo: str, commit: str) -> None:
    if ws.exists():
        shutil.rmtree(ws, onexc=lambda f, p, e: (Path(p).chmod(0o700), f(p)))
    ws.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=false", repo, str(ws)], check=True)
    git(ws, "checkout", "-q", "-B", "baton", commit)
    git(ws, "config", "user.name", "baton")
    git(ws, "config", "user.email", "baton@localhost")


def fingerprint(ws: Path) -> str:
    """Hash of every staged path and its content, ignoring file modes, so Windows and Linux agree."""
    tree = git(ws, "write-tree").decode().strip()
    listing = git(ws, "ls-tree", "-r", "-z", tree).split(b"\0")
    digest = hashlib.sha256()
    for entry in filter(None, listing):
        meta, path = entry.split(b"\t", 1)
        digest.update(path + b"\0" + meta.split()[2] + b"\n")
    return digest.hexdigest()[:16]


def staged_diff(ws: Path) -> str | None:
    git(ws, "add", "-A")
    diff = git(ws, "diff", "--cached", "--binary", "HEAD")
    return base64.b64encode(diff).decode() if diff else None


def commit(ws: Path, message: str) -> None:
    git(ws, "add", "-A")
    git(ws, "commit", "-q", "--allow-empty", "-m", message)


def reset(ws: Path) -> None:
    git(ws, "reset", "-q", "--hard", "HEAD")
    git(ws, "clean", "-q", "-fd")


def apply(ws: Path, diff_b64: str, message: str) -> None:
    git(ws, "apply", "--index", "--binary", "-", stdin=base64.b64decode(diff_b64))
    git(ws, "commit", "-q", "-m", message)
