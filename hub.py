import argparse
import asyncio
import io
import json
import sqlite3
import time
import zipfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

HERE = Path(__file__).parent
LEASE_TTL = 30.0
MAX_EVENT_BYTES = 512 * 1024
SLOW_CLIENT_BACKLOG = 2000
MEMBER_TYPES = {"task", "steer", "takeover"}
AGENT_TYPES = {"turn", "result", "done"}
HARNESS_FILES = ["harness.py", "render.py", "workspace.py", "requirements.txt"]

SCHEMA = """
create table if not exists events (
  seq integer primary key autoincrement,
  id text unique not null,
  ts real not null,
  actor text not null,
  type text not null,
  epoch integer,
  payload text not null
);
create table if not exists lease (
  k integer primary key check (k = 1),
  holder text,
  epoch integer not null,
  expires_at real not null
);
insert or ignore into lease values (1, null, 0, 0);
"""


class Hub:
    def __init__(self, db_path: str, participants: dict[str, dict]):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma journal_mode=wal")
        self.db.executescript(SCHEMA)
        self.participants = participants
        self.lock = asyncio.Lock()
        self.clients: dict[asyncio.Queue, str] = {}

    def rows_after(self, since: int) -> list[dict]:
        cur = self.db.execute("select * from events where seq > ? order by seq", (since,))
        return [self._row(r) for r in cur]

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {**dict(r), "payload": json.loads(r["payload"])}

    def lease(self) -> dict:
        r = self.db.execute("select holder, epoch, expires_at from lease").fetchone()
        live = r["holder"] is not None and r["expires_at"] > time.time()
        return {"holder": r["holder"] if live else None, "epoch": r["epoch"],
                "expires_in": round(max(0.0, r["expires_at"] - time.time()), 1) if live else 0}

    def _store(self, event_id: str, actor: str, type_: str, epoch: int | None, payload: dict) -> dict:
        cur = self.db.execute(
            "insert into events (id, ts, actor, type, epoch, payload) values (?, ?, ?, ?, ?, ?)",
            (event_id, time.time(), actor, type_, epoch, json.dumps(payload)))
        self.db.commit()
        row = self._row(self.db.execute("select * from events where seq = ?", (cur.lastrowid,)).fetchone())
        self._broadcast({"op": "event", **row})
        return row

    def _broadcast(self, message: dict) -> None:
        for queue in list(self.clients):
            if queue.qsize() > SLOW_CLIENT_BACKLOG:
                self.clients.pop(queue)
                queue.put_nowait(None)
            else:
                queue.put_nowait(message)

    def _presence(self) -> None:
        self._broadcast({"op": "presence", "names": sorted(set(self.clients.values()))})

    def _set_lease(self, holder: str | None, epoch: int, reason: str, previous: str | None) -> None:
        expires_at = time.time() + LEASE_TTL if holder else 0
        self.db.execute("update lease set holder = ?, epoch = ?, expires_at = ?", (holder, epoch, expires_at))
        self.db.commit()
        self._store(f"lease-{epoch}-{reason}", "hub", "lease", epoch,
                    {"holder": holder, "previous": previous, "reason": reason})
        self._broadcast({"op": "lease", **self.lease()})

    async def join(self, name: str, queue: asyncio.Queue, since: int) -> None:
        async with self.lock:
            for row in self.rows_after(since):
                queue.put_nowait({"op": "event", **row})
            queue.put_nowait({"op": "lease", **self.lease()})
            self.clients[queue] = name
            self._presence()

    async def leave(self, queue: asyncio.Queue) -> None:
        async with self.lock:
            if self.clients.pop(queue, None) is not None:
                self._presence()

    async def handle(self, who: dict, msg: dict) -> dict | None:
        op, name = msg.get("op"), who["name"]
        if op == "ping":
            return {"op": "pong"}
        if who["role"] != "member":
            return {"op": "nack", "id": msg.get("id"), "reason": "viewers are read-only"}
        async with self.lock:
            if op == "append":
                return self._append(name, msg)
            if op == "acquire":
                return self._acquire(name)
            if op == "renew":
                return self._renew(name, msg.get("epoch"))
            if op == "release":
                return self._release(name, msg.get("epoch"))
        return {"op": "nack", "reason": f"unknown op {op!r}"}

    def _append(self, name: str, msg: dict) -> dict:
        event_id, type_, payload = msg.get("id"), msg.get("type"), msg.get("payload")
        if not isinstance(event_id, str) or not 0 < len(event_id) <= 100 or not isinstance(payload, dict):
            return {"op": "nack", "id": event_id, "reason": "append needs a string id and an object payload"}
        existing = self.db.execute("select seq from events where id = ?", (event_id,)).fetchone()
        if existing:
            return {"op": "ack", "id": event_id, "seq": existing["seq"], "duplicate": True}
        if len(json.dumps(payload)) > MAX_EVENT_BYTES:
            return {"op": "nack", "id": event_id, "reason": f"event larger than {MAX_EVENT_BYTES} bytes"}
        epoch = None
        if type_ in AGENT_TYPES:
            lease = self.lease()
            if lease["holder"] != name:
                return {"op": "nack", "id": event_id, "reason": f"not the driver (driver: {lease['holder']})"}
            if msg.get("epoch") != lease["epoch"]:
                return {"op": "nack", "id": event_id, "reason": f"stale epoch {msg.get('epoch')} (current {lease['epoch']})"}
            epoch = lease["epoch"]
        elif type_ not in MEMBER_TYPES:
            return {"op": "nack", "id": event_id, "reason": f"unknown event type {type_!r}"}
        elif type_ == "task" and self.db.execute("select 1 from events where type = 'task'").fetchone():
            return {"op": "nack", "id": event_id, "reason": "this session already has its task"}
        row = self._store(event_id, name, type_, epoch, payload)
        return {"op": "ack", "id": event_id, "seq": row["seq"]}

    def _acquire(self, name: str) -> dict:
        lease = self.lease()
        if lease["holder"] == name:
            return {"op": "acquired", **lease}
        if lease["holder"] is not None:
            return {"op": "nack", "reason": f"{lease['holder']} is driving ({lease['expires_in']}s left)"}
        previous = self.db.execute("select holder from lease").fetchone()["holder"]
        self._set_lease(name, lease["epoch"] + 1, "expired" if previous else "free", previous)
        return {"op": "acquired", **self.lease()}

    def _renew(self, name: str, epoch) -> dict:
        lease = self.lease()
        if lease["holder"] != name or lease["epoch"] != epoch:
            return {"op": "lost", **lease}
        self.db.execute("update lease set expires_at = ?", (time.time() + LEASE_TTL,))
        self.db.commit()
        return {"op": "renewed", **self.lease()}

    def _release(self, name: str, epoch) -> dict:
        lease = self.lease()
        if lease["holder"] != name or lease["epoch"] != epoch:
            return {"op": "lost", **lease}
        self._set_lease(None, lease["epoch"], "released", name)
        return {"op": "released", **self.lease()}


def create_app(hub: Hub) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def page():
        return FileResponse(HERE / "page.html")

    @app.get("/harness.zip")
    async def harness_zip():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for name in HARNESS_FILES:
                if (HERE / name).exists():
                    z.write(HERE / name, name)
        return Response(buffer.getvalue(), media_type="application/zip")

    @app.get("/health")
    async def health():
        return {"ok": True, "lease": hub.lease(), "clients": len(hub.clients)}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket, token: str = ""):
        who = hub.participants.get(token)
        if who is None:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()

        async def pump():
            try:
                while (message := await queue.get()) is not None:
                    await websocket.send_json(message)
                await websocket.close(code=4408, reason="too far behind; reconnect with hello")
            except (WebSocketDisconnect, RuntimeError):
                pass

        sender = asyncio.create_task(pump())
        try:
            queue.put_nowait({"op": "welcome", **who})
            async for msg in websocket.iter_json():
                if not isinstance(msg, dict):
                    continue
                if msg.get("op") == "hello":
                    await hub.join(who["name"], queue, int(msg.get("since", 0)))
                    continue
                reply = await hub.handle(who, msg)
                if reply is not None:
                    queue.put_nowait(reply)
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        finally:
            await hub.leave(queue)
            sender.cancel()

    return app


def main() -> None:
    global LEASE_TTL
    parser = argparse.ArgumentParser(description="Baton hub: orders the log and holds the driver lease")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--db", default="baton.db")
    parser.add_argument("--participants", default="participants.json")
    parser.add_argument("--lease-ttl", type=float, default=LEASE_TTL)
    args = parser.parse_args()
    LEASE_TTL = args.lease_ttl
    participants = json.loads(Path(args.participants).read_text())
    uvicorn.run(create_app(Hub(args.db, participants)), host="0.0.0.0", port=args.port, ws_ping_interval=20)


if __name__ == "__main__":
    main()
