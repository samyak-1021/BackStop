#!/usr/bin/env python3
"""Render the headline curves from results.json as committed SVGs.

    python scripts/make_chart.py

Writes ``results/curve-light.svg`` and ``results/curve-dark.svg``. The README
selects between them with a ``<picture>`` element, which is how GitHub does
theme-aware images — a single SVG with an internal media query is not reliably
honoured there.

Two panels rather than one chart with four lines. Correctness and orphan rate
are both percentages, so they *could* share an axis, but they answer different
questions and overlaying them would force the reader to untangle which line
belongs to which. Small multiples keep each comparison to two series.

Colours are the validated categorical slots 1 and 2, checked with the palette
validator for both surfaces (adjacent-pair ΔE 33.6 normal / 24.7 protan on
light; 31.8 / 26.8 on dark). Identity is never colour-alone: every line is
directly labelled.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results/results.json")

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#8a8880",
        "grid": "#e5e4df",
        "axis": "#c9c7c0",
        "runtime": "#2a78d6",
        "baseline": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#8a8880",
        "grid": "#2e2e2c",
        "axis": "#454542",
        "runtime": "#3987e5",
        "baseline": "#d95926",
    },
}

WIDTH = 880
HEIGHT = 360
PAD_L = 54
PAD_R = 18
PAD_T = 54
PAD_B = 52
GAP = 56
PANEL_W = (WIDTH - PAD_L - PAD_R - GAP) // 2
PANEL_H = HEIGHT - PAD_T - PAD_B


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_curve() -> tuple[list[float], dict[str, list[float]], dict[str, list[float]]]:
    data = json.loads(RESULTS.read_text())
    rates = sorted({row["fault_rate"] for row in data["curve"]})
    correct: dict[str, list[float]] = {"baseline": [], "runtime": []}
    orphans: dict[str, list[float]] = {"baseline": [], "runtime": []}
    for rate in rates:
        for label in ("baseline", "runtime"):
            row = next(
                r
                for r in data["curve"]
                if r["fault_rate"] == rate and r["label"] == label
            )
            correct[label].append(row["correct_rate"] * 100)
            orphans[label].append(row["orphan_rate"] * 100)
    return rates, correct, orphans


def panel(
    x0: int,
    title: str,
    subtitle: str,
    rates: list[float],
    series: dict[str, list[float]],
    theme: dict[str, str],
    label_side: dict[str, str],
) -> str:
    """One small multiple: two series over the fault-rate axis."""
    parts: list[str] = []
    y0 = PAD_T
    max_rate = max(rates)

    def px(rate: float) -> float:
        return x0 + (rate / max_rate) * PANEL_W

    def py(value: float) -> float:
        return y0 + PANEL_H - (value / 100.0) * PANEL_H

    parts.append(
        f'<text x="{x0}" y="{y0 - 28}" font-size="14" font-weight="600" '
        f'fill="{theme["text_primary"]}">{esc(title)}</text>'
    )
    parts.append(
        f'<text x="{x0}" y="{y0 - 11}" font-size="11.5" '
        f'fill="{theme["text_muted"]}">{esc(subtitle)}</text>'
    )

    # Recessive gridlines, labelled on the left panel only to avoid repetition.
    for value in (0, 25, 50, 75, 100):
        y = py(value)
        parts.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + PANEL_W}" y2="{y:.1f}" '
            f'stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        if x0 == PAD_L:
            parts.append(
                f'<text x="{x0 - 10}" y="{y + 4:.1f}" font-size="11" '
                f'text-anchor="end" fill="{theme["text_muted"]}">{value}%</text>'
            )

    parts.append(
        f'<line x1="{x0}" y1="{y0 + PANEL_H}" x2="{x0 + PANEL_W}" '
        f'y2="{y0 + PANEL_H}" stroke="{theme["axis"]}" stroke-width="1"/>'
    )

    for rate in rates:
        x = px(rate)
        parts.append(
            f'<text x="{x:.1f}" y="{y0 + PANEL_H + 18}" font-size="11" '
            f'text-anchor="middle" fill="{theme["text_muted"]}">'
            f"{int(rate * 100)}%</text>"
        )

    # 2px lines, >=8px markers, and a surface ring so overlapping marks stay
    # readable where the two series touch.
    for name in ("baseline", "runtime"):
        colour = theme[name]
        points = " ".join(
            f"{px(r):.1f},{py(v):.1f}" for r, v in zip(rates, series[name], strict=True)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for rate, value in zip(rates, series[name], strict=True):
            parts.append(
                f'<circle cx="{px(rate):.1f}" cy="{py(value):.1f}" r="4" '
                f'fill="{colour}" stroke="{theme["surface"]}" stroke-width="2"/>'
            )

        # Direct label at the end of the line: identity never rests on colour.
        last_x = px(rates[-1])
        last_y = py(series[name][-1])
        side = label_side[name]
        dy = -12 if side == "above" else 18
        parts.append(
            f'<text x="{last_x:.1f}" y="{last_y + dy:.1f}" font-size="11.5" '
            f'font-weight="600" text-anchor="end" fill="{theme["text_secondary"]}">'
            f"{esc(name)}</text>"
        )

    return "\n  ".join(parts)


def render(theme_name: str) -> str:
    theme = THEMES[theme_name]
    rates, correct, orphans = load_curve()

    body = [
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{theme["surface"]}"/>',
        f'<text x="{PAD_L}" y="26" font-size="15.5" font-weight="700" '
        f'fill="{theme["text_primary"]}">Behaviour as the tools get less '
        f"reliable</text>",
        panel(
            PAD_L,
            "Correctness",
            "did the right thing — higher is better",
            rates,
            correct,
            theme,
            {"runtime": "above", "baseline": "below"},
        ),
        panel(
            PAD_L + PANEL_W + GAP,
            "Orphan rate",
            "world left inconsistent — lower is better",
            rates,
            orphans,
            theme,
            {"runtime": "below", "baseline": "above"},
        ),
        f'<text x="{PAD_L}" y="{HEIGHT - 12}" font-size="11" '
        f'fill="{theme["text_muted"]}">fault rate — share of tool calls that '
        f"misbehave · 200 episodes per point</text>",
    ]

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'font-family="ui-sans-serif, -apple-system, Segoe UI, Helvetica, '
        f'Arial, sans-serif" role="img" '
        f'aria-label="Two line charts. Correctness falls from 100 percent to '
        f'about 17 percent for the baseline as the fault rate rises, while the '
        f'runtime stays at or near 100 percent until 45 percent faults. Orphan '
        f'rate rises to over half of episodes for the baseline and stays near '
        f'zero for the runtime.">\n  '
        + "\n  ".join(body)
        + "\n</svg>\n"
    )


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit("run scripts/run_sweep.py first — results.json is missing")
    for name in THEMES:
        out = Path(f"results/curve-{name}.svg")
        out.write_text(render(name))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
