import json

SYSTEM = """You are a coding agent working inside a shared session. Several people and several models may have taken the earlier steps; messages from people start with their name, and notes about the session start with "Session:". Work only inside the workspace, using the tools. Commands run in bash with a 60-second limit, so never start servers or anything that keeps running. Run Python as `python`, never `python3`. Change code with write_file, giving the whole new file. When the task is done, reply without calling a tool and say briefly what you changed."""

RESULT_CHARS = 4000
DIGEST_RESULT_CHARS = 200


def estimate_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


def _blocks(events: list[dict]) -> list[list[dict]]:
    """Group events so an assistant turn always stays together with the results of its tool calls."""
    blocks: list[list[dict]] = []
    for e in events:
        if e["type"] == "result" and e["payload"].get("call") is not None and blocks and blocks[-1][0]["type"] == "turn":
            blocks[-1].append(e)
        else:
            blocks.append([e])
    return blocks


def _messages(block: list[dict]) -> list[dict]:
    head, p = block[0], block[0]["payload"]
    if head["type"] == "steer":
        return [{"role": "user", "content": f"{head['actor']}: {p['text']}"}]
    if head["type"] == "lease":
        return [{"role": "user", "content": f"Session: {p['holder']} is now driving."}] if p.get("holder") else []
    if head["type"] == "result":
        return [{"role": "user", "content": f"Session: setup `{p['args'].get('command')}` finished:\n{p['output'][:RESULT_CHARS]}"}]
    if head["type"] != "turn":
        return []
    calls = p.get("tool_calls", [])
    out = [{"role": "assistant", "content": p.get("text", ""),
            "tool_calls": [{"function": {"name": c["name"], "arguments": c["args"]}} for c in calls]}]
    results = {r["payload"]["call"]: r["payload"] for r in block[1:]}
    for i, c in enumerate(calls):
        r = results.get(i)
        content = r["output"][:RESULT_CHARS] if r else "Not run: the driver changed before this call ran."
        out.append({"role": "tool", "tool_name": c["name"], "content": content})
    return out


def _digest(blocks: list[list[dict]]) -> dict:
    lines = []
    for block in blocks:
        head, p = block[0], block[0]["payload"]
        if head["type"] == "steer":
            lines.append(f"- {head['actor']} said: {p['text']}")
        elif head["type"] == "turn":
            by = p.get("model", "?")
            for r in block[1:]:
                rp = r["payload"]
                args = json.dumps(rp["args"])[:120]
                lines.append(f"- [{by}] {rp['name']}({args}) -> {rp['output'][:DIGEST_RESULT_CHARS]!r}")
    return {"role": "user", "content": "Session: summary of earlier steps, built by rule:\n" + "\n".join(lines)}


def render(events: list[dict], budget_tokens: int) -> list[dict]:
    task = next(e for e in events if e["type"] == "task")
    fixed = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"{task['actor']} set the task: {task['payload']['text']}"}]
    blocks = _blocks([e for e in events if e["seq"] > task["seq"] and e["type"] != "task"])
    limit = int(budget_tokens * 0.75)
    recent: list[dict] = []
    cut = len(blocks)
    while cut > 0:
        candidate = _messages(blocks[cut - 1]) + recent
        if estimate_tokens(fixed + candidate) > limit:
            break
        recent, cut = candidate, cut - 1
    older = [_digest(blocks[:cut])] if cut else []
    return fixed + older + recent
