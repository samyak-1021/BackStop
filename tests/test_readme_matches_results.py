"""The README is checked against the data it claims to report.

A results table in a README is a promise that some program produced those
numbers. The promise decays: an experiment is re-run, a bug is fixed, the
figures shift, and the prose keeps saying what it said in the first draft. By
then the document is confidently wrong and nothing fails.

So the README is parsed here and every number in it is looked up in
``results/results.json``. Change the code, re-run the sweep, and any figure the
prose still asserts but the data no longer supports becomes a failing test with
the stale value printed next to the real one.

This is deliberately strict about the *published* claims only. Rounding is
allowed to half a point, because the README quotes one decimal place.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "results" / "results.json"

# Where the README's wording for a row differs from the label the sweep writes
# into results.json. Tried only after the literal label misses, because the same
# word can mean different rows in different sections: "baseline" is a label of
# its own in the curve, and shorthand for "none (baseline)" in the ablation.
ALIASES = {
    "baseline": "none (baseline)",
    "+ compensation and reconciliation": "+ compensation+reconcile",
    "respects the point of no return": "respects the PONR",
    "+ compensation, unwinding blindly": "+ compensation, unwinding blindly",
}

TOLERANCE = 0.005  # half a percentage point — the README quotes one decimal


@pytest.fixture(scope="module")
def data() -> dict:
    if not RESULTS.exists():  # pragma: no cover - only in a broken checkout
        pytest.skip("results.json missing; run scripts/run_sweep.py")
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


# --- Parsing -----------------------------------------------------------------


def clean(cell: str) -> str:
    """Strip the markdown emphasis the README uses to highlight a row."""
    return re.sub(r"[*`]", "", cell).strip()


def tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown table in the document, as (header, rows)."""
    out: list[tuple[list[str], list[list[str]]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines):
            separator = lines[i + 1].strip()
            if re.fullmatch(r"\|[\s\-:|]+\|", separator):
                header = [clean(c) for c in lines[i].strip().strip("|").split("|")]
                rows = []
                j = i + 2
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    rows.append(
                        [clean(c) for c in lines[j].strip().strip("|").split("|")]
                    )
                    j += 1
                out.append((header, rows))
                i = j
                continue
        i += 1
    return out


def table_with(text: str, *header_fragments: str) -> list[list[str]]:
    """The rows of the one table whose header contains all these fragments."""
    matches = [
        rows
        for header, rows in tables(text)
        if all(any(f in h for h in header) for f in header_fragments)
    ]
    assert len(matches) == 1, (
        f"expected exactly one table with headers {header_fragments}, "
        f"found {len(matches)}"
    )
    return matches[0]


def pct(cell: str) -> float:
    return float(clean(cell).rstrip("%")) / 100.0


def rate(cell: str) -> float:
    return round(float(clean(cell).rstrip("%")) / 100.0, 4)


# --- Lookups -----------------------------------------------------------------


def find(data: dict, section: str, label: str, fault_rate: float) -> dict:
    """Resolve a README row label to its results.json row.

    Three spellings are tried, in order of how much they assume:

    1. The label exactly as written.
    2. Its alias, where the README's wording differs from the sweep's.
    3. The label with a trailing parenthetical removed — a README row may
       annotate itself ("+ compensation, unwinding blindly (the old
       behaviour)") and that annotation is for the reader, not part of the
       identifier.

    Order matters. Stripping first turns the real label "none (baseline)" into
    "none" and loses it, which is how the first version of this failed.
    """
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    candidates = (label, ALIASES.get(label), stripped, ALIASES.get(stripped))

    for wanted in candidates:
        if wanted is None:
            continue
        for row in data[section]:
            if row["label"] == wanted and abs(row["fault_rate"] - fault_rate) < 1e-9:
                return row
    raise AssertionError(
        f"no {section} row for {label!r} at fault rate {fault_rate}; "
        f"have {sorted({(r['label'], r['fault_rate']) for r in data[section]})}"
    )


def close(claimed: float, actual: float, what: str) -> None:
    assert abs(claimed - actual) <= TOLERANCE, (
        f"README says {what} = {claimed:.1%}, results.json says {actual:.1%}"
    )


# --- The published tables ----------------------------------------------------


def test_headline_curve_matches_results(data: dict, readme: str) -> None:
    rows = table_with(readme, "Fault rate", "Baseline correct", "Calls")
    assert len(rows) == len(data["fault_rates"]), "a fault rate went missing"

    for row in rows:
        r = rate(row[0])
        base = find(data, "curve", "baseline", r)
        run = find(data, "curve", "runtime", r)

        close(pct(row[1]), base["correct_rate"], f"baseline correct @ {r}")
        close(pct(row[2]), base["orphan_rate"], f"baseline orphans @ {r}")
        close(pct(row[3]), run["correct_rate"], f"runtime correct @ {r}")
        close(pct(row[4]), run["orphan_rate"], f"runtime orphans @ {r}")

        # "6 → 12": both arms' median call counts. Publishing only the runtime's
        # left the reader unable to see what the reliability cost, so the cheaper
        # column is asserted too.
        base_calls, run_calls = (
            int(x.strip()) for x in row[5].replace("->", "→").split("→")
        )
        assert base_calls == base["median_tool_calls"], f"baseline calls @ {r}"
        assert run_calls == run["median_tool_calls"], f"runtime calls @ {r}"
        assert int(row[6]) == run["median_retries"], f"median retries @ {r}"


def test_pass_k_table_matches_results(data: dict, readme: str) -> None:
    rows = table_with(readme, "pass^1", "pass^3", "pass^5")
    scores = {(row["label"], row["k"]): row["score"] for row in data["pass_k"]}

    for row in rows:
        label = row[0].lower()
        for k, cell in zip((1, 3, 5), row[1:], strict=True):
            close(pct(cell), scores[(label, k)], f"{label} pass^{k}")


def test_retry_amplification_table_matches_results(data: dict, readme: str) -> None:
    rows = table_with(readme, "Config (fault rate 60%)", "Double charges")

    for row in rows:
        cell = find(data, "ablation", row[0], 0.60)
        close(pct(row[1]), cell["correct_rate"], f"{row[0]} correct @ 0.6")
        close(pct(row[2]), cell["orphan_rate"], f"{row[0]} orphans @ 0.6")

        claimed = int(row[3])
        actual = cell["orphan_kinds"].get("double_charge", 0)
        assert claimed == actual, (
            f"README says {row[0]} caused {claimed} double charges, "
            f"results.json says {actual}"
        )


def test_point_of_no_return_table_matches_results(data: dict, readme: str) -> None:
    rows = table_with(readme, "Fault rate", "False notifications")

    for row in rows:
        r = rate(row[0])
        cell = find(data, "point_of_no_return", row[1], r)
        close(pct(row[2]), cell["correct_rate"], f"{row[1]} correct @ {r}")
        close(pct(row[3]), cell["orphan_rate"], f"{row[1]} orphans @ {r}")

        claimed = int(row[4])
        actual = cell["orphan_kinds"].get("false_notification", 0)
        assert claimed == actual, (
            f"README says {row[1]} @ {r} left {claimed} false notifications, "
            f"results.json says {actual}"
        )


def test_reconciliation_table_matches_results(data: dict, readme: str) -> None:
    rows = table_with(readme, "Orphans @ 20%", "Orphans @ 60%")

    for row in rows:
        for column, r in ((1, 0.20), (2, 0.60)):
            cell = find(data, "ablation", row[0], r)
            close(pct(row[column]), cell["orphan_rate"], f"{row[0]} orphans @ {r}")


def test_ordering_table_matches_results(data: dict, readme: str) -> None:
    rows = table_with(readme, "Policy", "Fault rate", "Correct", "Orphans")

    for row in rows:
        r = rate(row[1])
        cell = find(data, "ordering", row[0], r)
        close(pct(row[2]), cell["correct_rate"], f"{row[0]} correct @ {r}")
        close(pct(row[3]), cell["orphan_rate"], f"{row[0]} orphans @ {r}")


# --- Claims made in prose rather than in a table -----------------------------


def test_the_superadditivity_arithmetic_holds(data: dict, readme: str) -> None:
    """'+3.5, +16.5, together +30' has to still be true of the ablation."""
    base = find(data, "ablation", "none (baseline)", 0.20)["correct_rate"]
    retries = find(data, "ablation", "+ retries only", 0.20)["correct_rate"]
    idem = find(data, "ablation", "+ idempotency only", 0.20)["correct_rate"]
    both = find(data, "ablation", "retries + idempotency", 0.20)["correct_rate"]

    claimed = re.search(
        r"retries alone buy\s+\+([\d.]+) points, idempotency alone buys "
        r"\+([\d.]+) — and together they buy \+(\d+)",
        readme,
    )
    assert claimed, "the superadditivity sentence was reworded; re-check it"
    a, b, c = (float(x) for x in claimed.groups())

    close(a / 100, retries - base, "retries-alone gain")
    close(b / 100, idem - base, "idempotency-alone gain")
    close(c / 100, both - base, "combined gain")
    assert both - base > (retries - base) + (idem - base), (
        "the claim is that the protections are superadditive; they no longer are"
    )


def test_the_outcome_mix_claims_hold(data: dict, readme: str) -> None:
    """'17 of 200 roll forward ... 3 escalate' comes from the curve rows."""
    claimed = re.search(
        r"fault rate (\d+) of (\d+) episodes roll \*forward\*.*?"
        r"at (\d+)%, (\d+) escalate",
        readme,
        re.S,
    )
    assert claimed, "the roll-forward / escalation sentence was reworded"
    rolled, episodes, esc_rate, escalated = (int(x) for x in claimed.groups())

    assert episodes == data["episodes_per_cell"]
    assert rolled == find(data, "curve", "runtime", 0.60)["rolled_forward"]
    assert escalated == find(data, "curve", "runtime", esc_rate / 100)["escalated"]


def test_the_eager_policy_score_is_flat_and_is_the_impossible_share(
    data: dict,
) -> None:
    """The claim that 15.5% *is* the impossible-order share, not a coincidence.

    If this ever diverges, the README's explanation of finding 7 is wrong even
    though its table would still be right — which is exactly the kind of error
    a numbers-only check would miss.
    """
    eager = [r for r in data["ordering"] if r["label"] == "notify before ship"]
    assert len(eager) >= 3, "the zero-fault control row is missing"
    assert len({r["correct_rate"] for r in eager}) == 1, (
        "the eager policy's score is no longer flat across fault rates, "
        f"so it is not purely a precondition refusal: {[r for r in eager]}"
    )

    row = eager[0]
    impossible = (row["episodes"] - row["satisfiable"]) / row["episodes"]
    assert abs(row["correct_rate"] - impossible) < 1e-9, (
        "the eager policy scores on something other than the impossible orders"
    )
    assert row["orphan_rate"] == 0.0, "a refused action should damage nothing"


def test_every_episode_count_quoted_in_the_readme_is_real(
    data: dict, readme: str
) -> None:
    for quoted in re.findall(r"(\d+) episodes per cell", readme):
        assert int(quoted) == data["episodes_per_cell"]
    for quoted in re.findall(r"per (\d+) episodes", readme):
        assert int(quoted) == data["episodes_per_cell"]


# --- The other direction ------------------------------------------------------
#
# Every test above starts from a README row and looks it up in results.json.
# That catches a stale number. It cannot catch a *missing* one: deleting the two
# least flattering rows from this document — the one where retries triple the
# double-charges, and the one where blind unwinding does the most damage — left
# the whole suite passing. A one-directional check is a check against typos, not
# against selective reporting.

# Sections whose every row must appear in the README: the header fragments that
# identify the table, and which of its columns hold the label and the fault rate.
# A section is listed here because it is *published*; leaving one out is a
# deliberate act with a reason attached.
#
# The columns are specified rather than searched for a reason. The first version
# of this test asked only "do this label and this rate appear anywhere in the
# document" — and deleting a point-of-no-return row sailed through, because the
# other rows of that same table still mention both. A presence check has to look
# at the table the row belongs to, not at the prose around it.
PUBLISHED_SECTIONS = {
    # label_col is None where the table has no label column: the curve's rows are
    # fault rates and both arms are columns, so presence means "this rate has a
    # row" and the headline test's length assertion covers the rest.
    "curve": {"headers": ("Fault rate", "Baseline correct", "Calls"),
              "label_col": None, "rate_col": 0},
    "point_of_no_return": {"headers": ("Fault rate", "False notifications"),
                           "label_col": 1, "rate_col": 0},
    "ordering": {"headers": ("Policy", "Fault rate", "Correct", "Orphans"),
                 "label_col": 0, "rate_col": 1},
}

# Ablation rows deliberately not shown, with the reason. Anything not listed
# here has to be in the README.
UNPUBLISHED_ABLATION_ROWS = {
    "+ validation only": "inert under the scripted policy; discussed in prose instead",
    "+ preconditions only": "inert under a well-behaved policy; discussed in prose",
    "everything (runtime)": "identical to the curve's runtime row at the same rate",
}


def _mentions(text: str, label: str) -> bool:
    """Is this results.json label referred to anywhere in the document?

    The README does not always use the sweep's internal label — it writes
    "baseline" for "none (baseline)", for instance — so every alias that maps to
    this label counts as a mention too. Without that, tightening the reverse
    check would just produce false alarms, and a check people learn to ignore is
    worse than no check.
    """
    if label in text:
        return True
    return any(
        wording in text for wording, canonical in ALIASES.items() if canonical == label
    )


def _canonical(label: str) -> str:
    """The results.json label a README cell is referring to."""
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    return ALIASES.get(label) or ALIASES.get(stripped) or label


def test_no_published_row_is_missing_from_the_readme(data: dict, readme: str) -> None:
    """Every row of every published section must appear in that section's table."""
    missing = []
    for section, spec in PUBLISHED_SECTIONS.items():
        rows = table_with(readme, *spec["headers"])

        published = set()
        for row in rows:
            fault_rate = rate(row[spec["rate_col"]])
            if spec["label_col"] is None:
                published.add((None, fault_rate))
            else:
                published.add((_canonical(row[spec["label_col"]]), fault_rate))
                # Also record the label as written, so a README that happens to
                # use the sweep's own spelling is not tripped up by aliasing.
                published.add((row[spec["label_col"]], fault_rate))

        for row in data[section]:
            key = (
                None if spec["label_col"] is None else row["label"],
                round(row["fault_rate"], 4),
            )
            if key not in published:
                missing.append(
                    f"{section}: {row['label']} @ {row['fault_rate']:.0%}"
                )

    assert not missing, (
        "results.json contains rows the README's own tables do not: "
        + "; ".join(sorted(set(missing)))
    )


def test_every_ablation_row_is_either_published_or_explained(
    data: dict, readme: str
) -> None:
    """The ablation is the easiest place to quietly drop an inconvenient row."""
    text = re.sub(r"[*`]", "", readme)

    silent = [
        row["label"]
        for row in data["ablation"]
        if row["label"] not in UNPUBLISHED_ABLATION_ROWS
        and not _mentions(text, row["label"])
    ]
    assert not silent, (
        "ablation row(s) neither shown in the README nor listed as deliberately "
        f"omitted: {sorted(set(silent))}"
    )

    stale = [
        label for label in UNPUBLISHED_ABLATION_ROWS
        if label not in {row["label"] for row in data["ablation"]}
    ]
    assert not stale, (
        f"the omission list names rows the sweep no longer produces: {stale}"
    )


async def test_the_published_numbers_are_reproducible_from_this_code() -> None:
    """The link the README↔results tests cannot make.

    `results.json` is a checked-in file. Everything above proves the README
    agrees with it; nothing proved *it* agrees with the code, so a stale
    results file would sail through. This re-runs one cell and compares.

    Deliberately small — 40 of the 200 published seeds — so it costs seconds
    rather than minutes. It is a tripwire for "the code moved and the numbers
    did not", not a re-derivation of the sweep.
    """
    from backstop.eval.sweep import run_cell
    from backstop.policies.scripted import ScriptedPolicy
    from backstop.runtime.engine import BASELINE, RUNTIME

    data = json.loads(RESULTS.read_text())
    seeds = range(40)

    for label, config in (("baseline", BASELINE), ("runtime", RUNTIME)):
        published = next(
            r
            for r in data["curve"]
            if r["label"] == label and r["fault_rate"] == 0.20
        )
        measured = await run_cell(label, ScriptedPolicy(), config, 0.20, seeds)

        # A 40-seed subsample of a 200-seed cell, so the tolerance is sampling
        # noise, not slack: 15 points would still catch a real regression while
        # tolerating the subsample.
        assert abs(measured.correct_rate - published["correct_rate"]) < 0.15, (
            f"{label} correctness has drifted from the published figure: "
            f"{measured.correct_rate:.1%} now vs {published['correct_rate']:.1%} "
            "in results.json — re-run scripts/run_sweep.py"
        )
        assert abs(measured.orphan_rate - published["orphan_rate"]) < 0.15, (
            f"{label} orphan rate has drifted: {measured.orphan_rate:.1%} now "
            f"vs {published['orphan_rate']:.1%} in results.json"
        )
