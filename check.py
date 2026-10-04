import argparse
import json
import statistics
from pathlib import Path

AGENT_TYPES = {"turn", "result", "done"}


def load(path: Path) -> list[dict]:
    """A harness restarted with the same --log appends to it; a new process always catches up from seq 1."""
    lives = []
    for text in path.read_text(encoding="utf-8").splitlines():
        if not text.strip():
            continue
        row = json.loads(text)
        if not lives or (row["kind"] == "event" and row["seq"] == 1):
            lives.append({"events": [], "rtt": [], "mirror": [], "refused": []})
        if row["kind"] == "event":
            lives[-1]["events"].append(row)
        elif row["kind"] == "rtt":
            lives[-1]["rtt"].append(row["ms"])
        elif row["kind"] in ("mirror", "refused"):
            lives[-1][row["kind"]].append(row)
    return lives


def line(ok: bool, name: str, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {detail}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check P1-P6 over the event logs every participant recorded")
    parser.add_argument("logs", nargs="+", type=Path)
    logs = {}
    for path in parser.parse_args().logs:
        for i, life in enumerate(load(path)):
            logs[path.stem if i == 0 else f"{path.stem}#{i + 1}"] = life
    results = []

    views = {name: {e["seq"]: e["id"] for e in log["events"]} for name, log in logs.items()}
    shared = set.intersection(*(set(v) for v in views.values()))
    agree = all(len({views[n][s] for n in views}) == 1 for s in shared)
    results.append(line(agree, "P1 one order", f"{len(logs)} participants agree on all {len(shared)} shared positions"))

    problems = []
    for name, log in logs.items():
        seqs = [e["seq"] for e in log["events"]]
        if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
            problems.append(f"{name} has gaps or repeats")
    results.append(line(not problems, "P6 no gaps, no repeats", "; ".join(problems) or "every log is contiguous"))

    events = max(logs.values(), key=lambda log: len(log["events"]))["events"]
    ids = [e["id"] for e in events]
    results.append(line(len(ids) == len(set(ids)), "P6 each event stored once", f"{len(ids)} events, {len(set(ids))} distinct ids"))

    holder, epoch, bad, periods = None, 0, [], []
    for e in events:
        if e["type"] == "lease":
            holder, epoch = e["payload"]["holder"], e["epoch"]
            if holder:
                periods.append({"epoch": epoch, "driver": holder, "models": set(), "turns": 0, "results": 0})
        elif e["type"] in AGENT_TYPES:
            if e["actor"] != holder or e["epoch"] != epoch:
                bad.append(e["seq"])
            elif e["type"] == "turn":
                periods[-1]["turns"] += 1
                periods[-1]["models"].add(e["payload"].get("model"))
            elif e["type"] == "result":
                periods[-1]["results"] += 1
    results.append(line(not bad, "P2 one driver", f"every agent event came from the lease holder at its epoch{'' if not bad else f'; violations at {bad}'}"))
    for name, log in logs.items():
        for r in log["refused"]:
            print(f"      fenced: {name}'s {r['type']} at epoch {r['epoch']} was refused ({r['reason']})")

    steers = [e["seq"] for e in events if e["type"] == "steer"]
    turns = [e for e in events if e["type"] == "turn"]
    delivered = [s for t in turns for s in t["payload"].get("steers", [])]
    last_turn = turns[-1]["seq"] if turns else 0
    due = [s for s in steers if s < last_turn]
    missing = [s for s in due if s not in delivered]
    twice = sorted({s for s in delivered if delivered.count(s) > 1})
    in_order = delivered == sorted(delivered)
    results.append(line(not missing and not twice and in_order, "P3 no lost redirect",
                        f"{len(due)} redirects due, {len(set(delivered))} delivered, missing {missing}, repeated {twice}, in order {in_order}"))

    mirrors = {name: log["mirror"] for name, log in logs.items() if log["mirror"]}
    checked = sum(len(m) for m in mirrors.values())
    matched = sum(r["match"] for m in mirrors.values() for r in m)
    results.append(line(checked > 0 and matched == checked, "P4 workspace travels",
                        f"{matched}/{checked} mirrored changes matched the driver's fingerprint ({', '.join(mirrors) or 'no followers'})"))

    models = {m for p in periods for m in p["models"] if m}
    done = [e for e in events if e["type"] == "done"]
    after = [p for p in periods[1:] if p["results"] or p["turns"]]
    results.append(line(len(models) > 1 and bool(after), "P5 model changes, run continues",
                        f"{len(periods)} driving periods over {len(models)} model families; later drivers took {sum(p['turns'] for p in after)} steps from the log"))
    print(f"      outcome: {'the driver declared the task done' if done else 'no driver declared the task done'} (whether the code works is checked separately)")
    for p in periods:
        print(f"      epoch {p['epoch']}: {p['driver']} on {', '.join(sorted(p['models'])) or '-'}: {p['turns']} turns, {p['results']} tool results")

    for name, log in logs.items():
        if log["rtt"]:
            print(f"      round trip for {name}: median {statistics.median(log['rtt']):.0f} ms, slowest {max(log['rtt']):.0f} ms over {len(log['rtt'])} sends")
    print(f"\n{sum(results)}/{len(results)} properties held")


if __name__ == "__main__":
    main()
