"""Render a small visually-rich corpus for experiment #7 (PIL — no matplotlib).

Each page is a PNG 'report figure' whose key fact is visible ONLY in the rendered
image (a bar height + printed value, or a table cell) — the value appears in NO
text document, so the text spine cannot retrieve it. This isolates the visual win:
both the parse-then-caption arm (VLM reads the figure) and the ColVision arm
(late-interaction over the page image) should find it; text cannot.

Usage:  uv run python scripts/gen_visual_corpus.py --out data/visual_corpus
Emits page PNGs + eval/golden/visual_golden.yaml (one query per planted fact).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_W, _H = 1024, 768


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT, size)


def _bar_chart(path: Path, title: str, bars: list[tuple[str, int]], unit: str) -> None:
    img = Image.new("RGB", (_W, _H), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), title, fill="black", font=_font(34))
    maxv = max(v for _, v in bars)
    x0, y0, chart_h, bw, gap = 120, 660, 480, 130, 60
    for i, (label, v) in enumerate(bars):
        h = int(chart_h * v / maxv)
        x = x0 + i * (bw + gap)
        d.rectangle([x, y0 - h, x + bw, y0], fill=(60, 110, 200))
        d.text((x, y0 + 12), label, fill="black", font=_font(22))
        d.text((x + 6, y0 - h - 34), f"{v:,}", fill="black", font=_font(26))
    d.text((40, 700), f"Values in {unit}.", fill=(80, 80, 80), font=_font(20))
    img.save(path)


def _table(path: Path, title: str, headers: list[str], rows: list[list[str]]) -> None:
    img = Image.new("RGB", (_W, _H), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), title, fill="black", font=_font(34))
    cols = len(headers)
    cw = (_W - 80) // cols
    y = 110
    for c, htxt in enumerate(headers):
        box = [40 + c * cw, y, 40 + (c + 1) * cw, y + 50]
        d.rectangle(box, outline="black", width=2, fill=(225, 230, 240))
        d.text((50 + c * cw, y + 12), htxt, fill="black", font=_font(24))
    y += 50
    for row in rows:
        for c, cell in enumerate(row):
            d.rectangle([40 + c * cw, y, 40 + (c + 1) * cw, y + 48], outline="black", width=1)
            d.text((50 + c * cw, y + 10), cell, fill="black", font=_font(22))
        y += 48
    img.save(path)


def _dense_table(path: Path, title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Caption-lossy page: ~30 small-font rows. A bounded VLM caption (≤300 tok) cannot
    transcribe every cell, so a query for a specific cell mismatches the caption — but
    ColVision's patch-level late-interaction can still locate it."""
    img = Image.new("RGB", (_W, _H), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 18), title, fill="black", font=_font(22))
    cw, rh, y0 = (_W - 60) // len(headers), 21, 70
    for c, htxt in enumerate(headers):
        d.rectangle([30 + c * cw, y0, 30 + (c + 1) * cw, y0 + 22], fill=(225, 230, 240))
        d.text((34 + c * cw, y0 + 3), htxt, fill="black", font=_font(14))
    for r, row in enumerate(rows):
        y = y0 + 22 + r * rh
        for c, cell in enumerate(row):
            d.rectangle([30 + c * cw, y, 30 + (c + 1) * cw, y + rh], outline=(180, 180, 180))
            d.text((34 + c * cw, y + 3), cell, fill="black", font=_font(14))
    img.save(path)


def _build_dense(out: Path) -> list[dict]:
    """Render dense pages + return their golden queries (one specific cell per page)."""
    (out / "pages").mkdir(parents=True, exist_ok=True)
    g: list[dict] = []

    parts = [[f"PX-{100 + i:03d}", f"{(i * 7 + 13) % 90:02d}", "ABCDE"[i % 5]] for i in range(30)]
    _dense_table(
        out / "pages" / "d1_parts.png",
        "Spare Parts Inventory (full)",
        ["Part", "Bin", "Aisle"],
        parts,
    )
    g.append(
        {
            "question": "Which bin holds spare part PX-117?",
            "page": "d1_parts.png",
            "answer": parts[17][1],
        }
    )

    servers = [[f"node-{i:02d}", f"10.4.{i}.{(i * 11) % 250}", f"{8000 + i}"] for i in range(30)]
    _dense_table(
        out / "pages" / "d2_servers.png",
        "Server Fleet — IP and Port Map",
        ["Host", "IP", "Port"],
        servers,
    )
    g.append(
        {
            "question": "What port does host node-21 listen on?",
            "page": "d2_servers.png",
            "answer": servers[21][2],
        }
    )

    skus = [
        [f"SKU-{200 + i:03d}", f"{(i * 37 + 5) % 500}", f"${(i * 3 + 9) % 90}.99"]
        for i in range(30)
    ]
    _dense_table(
        out / "pages" / "d3_stock.png", "Warehouse Stock Levels", ["SKU", "On hand", "Price"], skus
    )
    g.append(
        {
            "question": "How many units of SKU-214 are on hand?",
            "page": "d3_stock.png",
            "answer": skus[14][1],
        }
    )

    return [{"question": e["question"], "page": e["page"], "answer": e["answer"]} for e in g]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/visual_corpus")
    ap.add_argument("--golden", default="eval/golden/visual_golden.yaml")
    ap.add_argument("--dense", action="store_true", help="also emit caption-lossy dense pages")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)

    # page → (renderer, golden question, answer keyword)
    _bar_chart(
        out / "pages" / "p1_regional_sales.png",
        "Regional Sales — Q3 2026",
        [("Northwind", 4820), ("Eastgate", 3150), ("Southport", 5640), ("Westvale", 2790)],
        "units",
    )
    _bar_chart(
        out / "pages" / "p2_power_draw.png",
        "Datacenter Power Draw by Hall (kW)",
        [("Hall A", 812), ("Hall B", 1140), ("Hall C", 967), ("Hall D", 1325)],
        "kilowatts",
    )
    _table(
        out / "pages" / "p3_spare_bins.png",
        "Spare Parts — Bin Locations",
        ["Part", "Bin", "Shelf"],
        [["RX-12", "77", "C"], ["RX-19", "43", "A"], ["RX-24", "61", "B"], ["RX-31", "29", "D"]],
    )
    # Distractor pages (no golden fact) — push the page count past k=5 so recall@5
    # discriminates, and give the visual arms plausible near-neighbours to confuse.
    _table(
        out / "pages" / "p4_menu.png",
        "Cafeteria Menu Rotation",
        ["Day", "Soup", "Main"],
        [["Mon", "Tomato", "Pasta"], ["Tue", "Lentil", "Curry"], ["Wed", "Pea", "Pie"]],
    )
    _bar_chart(
        out / "pages" / "p5_headcount.png",
        "Headcount by Department",
        [("Eng", 142), ("Sales", 88), ("Ops", 51), ("HR", 23)],
        "people",
    )
    _bar_chart(
        out / "pages" / "p6_uptime.png",
        "Service Uptime by Cluster (%)",
        [("East", 99), ("West", 97), ("North", 98), ("South", 96)],
        "percent",
    )
    _table(
        out / "pages" / "p7_rooms.png",
        "Meeting Room Capacity",
        ["Room", "Seats", "Floor"],
        [["Aspen", "8", "2"], ["Birch", "14", "3"], ["Cedar", "6", "1"]],
    )
    _bar_chart(
        out / "pages" / "p8_tickets.png",
        "Support Tickets by Severity",
        [("P1", 12), ("P2", 47), ("P3", 130), ("P4", 64)],
        "tickets",
    )
    _table(
        out / "pages" / "p9_shifts.png",
        "On-call Shift Calendar",
        ["Week", "Primary", "Backup"],
        [["W1", "Avery", "Blake"], ["W2", "Casey", "Drew"], ["W3", "Erin", "Fran"]],
    )

    golden = {
        "dataset": {"name": "visual-golden", "version": "v1"},
        "queries": [
            {
                "question": "How many units did the Northwind region sell in Q3?",
                "page": "p1_regional_sales.png",
                "answer": "4820",
            },
            {
                "question": "What is the power draw of Hall B in the datacenter?",
                "page": "p2_power_draw.png",
                "answer": "1140",
            },
            {
                "question": "Which bin holds spare part RX-19?",
                "page": "p3_spare_bins.png",
                "answer": "43",
            },
            {
                "question": "What is the Q3 sales figure for the Southport region?",
                "page": "p1_regional_sales.png",
                "answer": "5640",
            },
        ],
    }
    if args.dense:
        golden["queries"] += _build_dense(out)

    gp = Path(args.golden)
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text(
        "# Visual golden (exp #7). Each answer is visible ONLY in the page image —\n"
        "# absent from all text docs, so the text spine cannot retrieve it.\n"
        + yaml.safe_dump(golden, sort_keys=False)
    )
    pages = sorted((out / "pages").glob("*.png"))
    print(f"Wrote {len(pages)} page images → {out}/pages")
    print(f"Wrote {len(golden['queries'])} golden queries → {gp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
