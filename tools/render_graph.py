# tools/render_graph.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph import build_graph  # noqa: E402

def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "mvp_graph.png"
    g = build_graph()
    png = g.get_graph().draw_mermaid_png()
    with open(out, "wb") as f:
        f.write(png)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
