import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import websockets
from ollama import AsyncClient
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

import workspace
from render import render

RENEW_EVERY = 5.0
REPLY_TIMEOUT = 10.0
MAX_STEPS = 30
OUTPUT_CHARS = 8000
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file in the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file with the full new content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a bash command in the workspace (60 s limit). Output is returned.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def short(args: dict) -> str:
    s = ", ".join(f"{k}={json.dumps(v)}" for k, v in (args or {}).items())
    return s if len(s) <= 80 else s[:80] + "..."


def describe(e: dict) -> str:
    p, t = e["payload"], e["type"]
    if t == "task":
        text = f"set the task: {p['text']}"
    elif t == "steer":
        text = f"redirected: {p['text']}"
    elif t == "takeover":
        text = f"asked to take over on {p.get('model') or 'their own model'}"
    elif t == "lease":
        text = f"{p['holder']} is now driving (epoch {e['epoch']})" if p["holder"] else f"{p['previous']} {p['reason']} the lease"
    elif t == "turn":
        if not p["tool_calls"] and not p.get("steers"):
            return ""
        text = f"[{p['model']}] " + ", ".join(f"{c['name']}({short(c['args'])})" for c in p["tool_calls"])
        if p.get("steers"):
            text += f"  (read redirects {', '.join(map(str, p['steers']))})"
    elif t == "result":
        lines = [line for line in p["output"].splitlines() if line.strip()] or [""]
        text = f"  -> {(lines[-1] if p['name'] == 'run' else lines[0])[:100]}"
        if p.get("diff"):
            text += f"  (file change, fingerprint {p['fingerprint']})"
    elif t == "done":
        text = f"[{p['model']}] finished: {(p['text'].strip().splitlines() or [''])[0]}"
    else:
        text = json.dumps(p)[:100]
    return f"#{e['seq']:<4}{e['actor']:<10}{text}"[:180]


class Harness:
    def __init__(self, args):
        self.args, self.url = args, args.hub.rstrip("/").replace("http", "ws", 1) + f"/ws?token={args.token}"
        self.name = None
        self.events: list[dict] = []
        self.pending: dict[str, asyncio.Future] = {}
        self.mine: dict[str, float] = {}
        self.produced: set[str] = set()
        self.handoff = self.thinking = False
        self.results_run = 0
        self.agent_model_call: asyncio.Task | None = None
        self.ws = None
        self.connected = asyncio.Event()
        self.want_drive = asyncio.Event()
        self.epoch = None
        self.agent: asyncio.Task | None = None
        self.bash = args.bash or shutil.which("bash")
        self.log = Path(args.log).open("a", encoding="utf-8") if args.log else None
        self.workspace = Path(args.workspace)
        self.synced_task = None

    # connection: reconnect forever, catch up from the last seq seen
    async def connection(self):
        delay = 1.0
        while True:
            try:
                async with websockets.connect(self.url, max_size=2**23) as ws:
                    self.ws, delay = ws, 1.0
                    await ws.send(json.dumps({"op": "hello", "since": self.events[-1]["seq"] if self.events else 0}))
                    self.connected.set()
                    pinger = asyncio.create_task(self.ping(ws))
                    try:
                        async for raw in ws:
                            await self.receive(json.loads(raw))
                    finally:
                        pinger.cancel()
            except (OSError, websockets.WebSocketException) as exc:
                print(f"[{self.name}] connection lost ({type(exc).__name__}); retrying in {delay:.0f}s")
            self.connected.clear()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15)

    async def ping(self, ws):
        while True:
            await asyncio.sleep(10)
            await ws.send(json.dumps({"op": "ping"}))
            if self.args.drop_after and time.time() % self.args.drop_after < 10:
                await ws.close()

    async def send(self, key: str, message: dict) -> dict:
        """Send and wait for the reply; on a timeout or a drop, resend the same message (the hub dedupes by id)."""
        while True:
            await self.connected.wait()
            future = self.pending[key] = asyncio.get_running_loop().create_future()
            try:
                await self.ws.send(json.dumps(message))
                if self.args.duplicate_sends and message.get("op") == "append":
                    await self.ws.send(json.dumps(message))
                return await asyncio.wait_for(future, REPLY_TIMEOUT)
            except (asyncio.TimeoutError, websockets.WebSocketException):
                continue

    async def append(self, type_: str, payload: dict, epoch: int | None = None) -> dict:
        event_id = f"{self.name}-{uuid.uuid4().hex[:12]}"
        self.mine[event_id] = time.perf_counter()
        self.produced.add(event_id)
        return await self.send(event_id, {"op": "append", "id": event_id, "type": type_, "payload": payload, "epoch": epoch})

    async def lease_op(self, op: str) -> dict:
        return await self.send("lease", {"op": op, "epoch": self.epoch})

    async def receive(self, m: dict):
        op = m["op"]
        if op == "welcome":
            self.name, self.role = m["name"], m["role"]
        elif op == "event":
            if self.events and m["seq"] <= self.events[-1]["seq"]:
                return
            self.events.append(m)
            self.record({"kind": "event", **m})
            if line := describe(m):
                print(line)
            if m["id"] in self.mine:
                self.record({"kind": "rtt", "ms": round((time.perf_counter() - self.mine.pop(m["id"])) * 1000, 1)})
            self.on_event(m)
        elif op in ("ack", "nack") and m.get("id") in self.pending:
            self.pending.pop(m["id"]).set_result(m)
        elif op in ("acquired", "renewed", "released", "lost", "nack") and "lease" in self.pending:
            self.pending.pop("lease").set_result(m)

    def record(self, row: dict):
        if self.log:
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()

    # follower side: mirror every change made by someone else
    def on_event(self, e: dict):
        p = e["payload"]
        if e["type"] == "task" and self.synced_task is None:
            self.synced_task = e
            workspace.prepare(self.workspace, p["repo"], p["commit"])
        elif e["type"] == "result" and p.get("diff") and e["id"] not in self.produced:
            self.mirror(e)
        elif e["type"] == "takeover":
            if e["actor"] == self.name:
                self.want_drive.set()
            elif self.agent and not self.agent.done():
                print(f"[{self.name}] {e['actor']} asked to take over; handing off at the next step boundary")
                self.handoff = True
                if self.thinking:
                    self.agent_model_call.cancel()

    def mirror(self, e: dict):
        try:
            workspace.apply(self.workspace, e["payload"]["diff"], f"seq {e['seq']}")
            mine = workspace.fingerprint(self.workspace)
        except RuntimeError as exc:
            mine = f"apply failed: {exc}"
        self.record({"kind": "mirror", "seq": e["seq"], "match": mine == e["payload"]["fingerprint"]})
        if mine != e["payload"]["fingerprint"]:
            print(f"[{self.name}] DIVERGED at seq {e['seq']} ({mine} != {e['payload']['fingerprint']}); resyncing from the log")
            self.resync()

    def resync(self):
        task = self.synced_task["payload"]
        workspace.prepare(self.workspace, task["repo"], task["commit"])
        for e in self.events:
            if e["type"] == "result" and e["payload"].get("diff"):
                workspace.apply(self.workspace, e["payload"]["diff"], f"seq {e['seq']}")

    # driver side
    async def drive_when_possible(self):
        while True:
            await self.want_drive.wait()
            reply = await self.lease_op("acquire")
            if reply["op"] != "acquired":
                await asyncio.sleep(2)
                continue
            self.epoch, self.handoff, self.thinking = reply["epoch"], False, False
            print(f"[{self.name}] driving at epoch {self.epoch} on {self.args.model}")
            renewer = asyncio.create_task(self.renew())
            self.agent = asyncio.create_task(self.agent_loop())
            try:
                await self.agent
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"[{self.name}] stopped driving: {type(exc).__name__}: {exc}")
            renewer.cancel()
            workspace.reset(self.workspace)
            if (await self.lease_op("release"))["op"] == "released":
                print(f"[{self.name}] released the lease")
            self.want_drive.clear()

    async def renew(self):
        while True:
            await asyncio.sleep(RENEW_EVERY)
            if (await self.lease_op("renew"))["op"] == "lost":
                print(f"[{self.name}] lease lost; discarding unaccepted changes and following")
                workspace.reset(self.workspace)
                self.agent.cancel()
                return

    async def agent_loop(self):
        await self.run_setup()
        cursor = max([s for e in self.events if e["type"] == "turn" for s in e["payload"].get("steers", [])], default=0)
        for _ in range(MAX_STEPS):
            if self.handoff:
                break
            steers = [e["seq"] for e in self.events if e["type"] == "steer" and e["seq"] > cursor]
            messages = render(self.events, self.args.num_ctx)
            self.thinking = True
            self.agent_model_call = asyncio.create_task(AsyncClient(host=self.args.ollama).chat(
                model=self.args.model, messages=messages, tools=TOOLS, think=False,
                options={"num_ctx": self.args.num_ctx, "temperature": 0.2}))
            try:
                response = await self.agent_model_call
            except asyncio.CancelledError:
                if self.handoff:
                    break
                raise
            finally:
                self.thinking = False
            msg = response.message
            calls = [{"name": c.function.name, "args": dict(c.function.arguments)} for c in (msg.tool_calls or [])]
            if not await self.accepted("turn", {"text": msg.content or "", "tool_calls": calls, "model": self.args.model, "steers": steers}):
                return
            cursor = max(steers, default=cursor)
            if not calls:
                await self.accepted("done", {"text": msg.content or "", "model": self.args.model})
                break
            for i, call in enumerate(calls):
                if not await self.tool_step(i, call["name"], call["args"]):
                    return

    async def run_setup(self):
        command = (self.synced_task or {}).get("payload", {}).get("install")
        if command:
            await self.tool_step(None, "run", {"command": command})

    async def tool_step(self, call: int | None, name: str, args: dict) -> bool:
        output = await asyncio.to_thread(self.run_tool, name, args)
        self.results_run += 1
        if self.results_run == self.args.freeze_after:
            print(f"[{self.name}] test T3: freezing the whole process for {self.args.freeze_seconds:.0f}s")
            time.sleep(self.args.freeze_seconds)
        diff = workspace.staged_diff(self.workspace)
        payload = {"call": call, "name": name, "args": args, "output": output[:OUTPUT_CHARS], "diff": diff,
                   "fingerprint": workspace.fingerprint(self.workspace) if diff else None}
        if await self.accepted("result", payload):
            if diff:
                workspace.commit(self.workspace, f"{name} by {self.name}")
            return True
        workspace.reset(self.workspace)
        return False

    async def accepted(self, type_: str, payload: dict) -> bool:
        reply = await self.append(type_, payload, epoch=self.epoch)
        if reply["op"] != "ack":
            print(f"[{self.name}] {type_} refused: {reply.get('reason')}")
            self.record({"kind": "refused", "type": type_, "epoch": self.epoch, "reason": reply.get("reason")})
        return reply["op"] == "ack"

    def run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "run":
                r = subprocess.run([self.bash, "-c", str(args["command"])], cwd=self.workspace,
                                   capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace")
                return f"exit {r.returncode}\n{r.stdout}{r.stderr}"
            path = (self.workspace / str(args["path"])).resolve()
            if self.workspace.resolve() not in path.parents or ".git" in path.parts:
                return f"error: {args['path']} is outside the workspace"
            if name == "read_file":
                return path.read_text(encoding="utf-8", errors="replace")
            if name == "write_file":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(str(args["content"]).encode("utf-8"))
                return f"wrote {len(str(args['content']))} characters to {args['path']}"
            return f"error: unknown tool {name}"
        except subprocess.TimeoutExpired:
            return "error: command timed out after 60 seconds"
        except KeyError as exc:
            expected = next(t["function"]["parameters"]["required"] for t in TOOLS if t["function"]["name"] == name)
            return f"error: {name} needs the arguments {expected}; missing {exc}"
        except OSError as exc:
            return f"error: {type(exc).__name__}: {exc}"

    async def check_model(self):
        try:
            pulled = {m.model for m in (await AsyncClient(host=self.args.ollama).list()).models}
        except ConnectionError as exc:
            raise SystemExit(f"cannot reach Ollama at {self.args.ollama}: {exc}")
        if self.args.model not in pulled and f"{self.args.model}:latest" not in pulled:
            raise SystemExit(f"model {self.args.model} is not pulled; run: ollama pull {self.args.model}")

    async def main(self):
        await self.check_model()
        asyncio.create_task(self.connection())
        await self.connected.wait()
        while self.name is None:
            await asyncio.sleep(0.05)
        if self.args.task:
            reply = await self.append("task", {"text": self.args.task, "repo": self.args.repo, "commit": self.args.commit, "install": self.args.install})
            if reply["op"] != "ack":
                raise SystemExit(f"task refused: {reply['reason']}")
            self.want_drive.set()
        if self.args.take_over:
            await asyncio.sleep(1)
            await self.append("takeover", {"model": self.args.model})
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            await self.drive_when_possible()
            return
        print(f"you are {self.name} ({self.role}) on {self.args.model} | type to redirect, /take to drive, /quit to leave")
        with patch_stdout():
            driver = asyncio.create_task(self.drive_when_possible())
            await self.terminal()
            driver.cancel()

    async def terminal(self):
        session = PromptSession()
        while True:
            try:
                text = (await session.prompt_async("› ")).strip()
            except (EOFError, KeyboardInterrupt):
                return
            if text == "/quit":
                return
            if not text:
                continue
            reply = await (self.append("takeover", {"model": self.args.model}) if text == "/take"
                           else self.append("steer", {"text": text}))
            if reply["op"] != "ack":
                print(f"refused: {reply.get('reason')}")


def cli():
    p = argparse.ArgumentParser(description="Baton harness: runs the agent when driving, mirrors the workspace when not")
    p.add_argument("--hub", required=True, help="hub address, e.g. http://127.0.0.1:8700")
    p.add_argument("--token", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--ollama", default="http://127.0.0.1:11434")
    p.add_argument("--workspace", default="workspaces/me")
    p.add_argument("--log", help="file that records every event received (read by check.py)")
    p.add_argument("--task", help="start a session with this task (makes this harness the first driver)")
    p.add_argument("--repo")
    p.add_argument("--commit")
    p.add_argument("--install", help="setup command a new driver runs, e.g. 'pip install -r requirements.txt'")
    p.add_argument("--take-over", action="store_true", help="ask the current driver to hand over")
    p.add_argument("--bash", help="path to bash (default: first bash on PATH)")
    p.add_argument("--drop-after", type=float, default=0, help="test T7: drop the connection about every N seconds")
    p.add_argument("--duplicate-sends", action="store_true", help="test T8: send every append twice")
    p.add_argument("--freeze-after", type=int, default=0, help="test T3: freeze the process after this many tool calls")
    p.add_argument("--freeze-seconds", type=float, default=45)
    asyncio.run(Harness(p.parse_args()).main())


if __name__ == "__main__":
    cli()
