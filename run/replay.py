"""Browse every eval in the viewer, and any two specs side by side: python run/replay.py"""

import argparse
import functools
import http.server
import json
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evals():
    """Newest first: every directory under runs/ that eval.py finished writing."""

    found = sorted(
        ROOT.glob("runs/*/evals/*/trajectory.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out = []
    for traj in found:
        d = traj.parent
        spec = json.loads((d / "eval.json").read_text(), parse_constant=lambda _: None)
        out.append(
            {
                "path": d.relative_to(ROOT).as_posix(),
                "run": d.parent.parent.name,
                "name": d.name,
                "summary": spec.get("summary", {}),
            }
        )
    return out


def specs():
    """Every task spec, then every paper spec."""

    tasks = sorted(ROOT.glob("research/notes/tasks/task_*.md"))
    papers = sorted(ROOT.glob("research/notes/literature/specs/*.md"))
    return [
        {"name": f.stem.removeprefix("task_"), "path": f.relative_to(ROOT).as_posix(),
         "task": f in tasks}
        for f in tasks + papers
        if f.name != "README.md"
    ]


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        lists = {"/evals.json": evals, "/specs.json": specs}
        if self.path not in lists:
            return super().do_GET()
        body = json.dumps(lists[self.path]()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    found = evals()
    if not found:
        raise SystemExit("no evals under runs/ — run run/eval.py first")

    url = f"http://localhost:{args.port}/tools/viewer/index.html"
    print(f"{len(found)} evals, newest {found[0]['path']}")
    print(f"{url}   (ctrl-c to stop)")
    print(f"specs side by side: http://localhost:{args.port}/tools/specs/index.html")
    webbrowser.open(url)

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Handler, directory=str(ROOT))
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
