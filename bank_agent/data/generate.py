"""Generate a synthetic, call-report-shaped sample dataset.

The real source for this project is the FDIC BankFind API (see ``fetch_fdic.py``).
This generator produces data with the same raw layout so the whole system runs
offline and in CI. Bank names are invented; the numbers are generated, not real.

The raw extract deliberately contains the kind of problems a real feed has:
duplicate re-submissions, numbers formatted with thousands separators, and a few
missing values. The silver layer is responsible for cleaning them.
"""
from __future__ import annotations

import csv
import random
from datetime import date
from pathlib import Path

STATES = ["MA", "CA", "NY", "TX", "FL", "IL", "WA", "CO", "GA", "NC", "OH", "AZ"]
TOWNS = [
    "Harbor Point", "Granite Falls", "Cedar Ridge", "Riverbend", "Maple Grove", "Stone Creek",
    "Bayview", "Pine Hollow", "Silver Lake", "Oak Valley", "Fairhaven", "North Fork",
    "Eastgate", "Willow Park", "Summit", "Lakeshore", "Brookfield", "Highland",
    "Prairie View", "Copper Hill",
]
SUFFIXES = ["Savings Bank", "Community Bank", "Bank & Trust", "National Bank", "Federal Savings", "State Bank"]
CHARTERS = ["N", "SM", "NM", "SB"]  # FDIC BKCLASS-style codes

QUARTER_ENDS = [
    date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2024, 12, 31),
    date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31),
]

INSTITUTION_FIELDS = ["CERT", "NAME", "CITY", "STALP", "BKCLASS"]
FINANCIAL_FIELDS = [
    "CERT", "REPDTE", "ASSET", "DEP", "LNLSNET", "NETINC", "EQ",
    "NET_INT_INC", "NONINT_INC", "NONINT_EXP", "EARNING_ASSETS",
    "TIER1_CAP", "RWA", "NONCURRENT_LOANS",
]


def _fmt(value: float, rng: random.Random) -> str:
    """Most numbers are plain integers; ~3% arrive with thousands separators."""
    v = int(round(value))
    return f"{v:,}" if rng.random() < 0.03 else str(v)


def generate(out_dir: Path, n_banks: int = 120, seed: int = 42) -> dict[str, int]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    names: set[str] = set()
    institutions = []
    for i in range(n_banks):
        while True:
            name = f"{rng.choice(TOWNS)} {rng.choice(SUFFIXES)}"
            if name not in names:
                names.add(name)
                break
        institutions.append({
            "CERT": 10000 + i * 7,
            "NAME": name,
            "CITY": name.split(" ")[0] if rng.random() < 0.5 else rng.choice(TOWNS),
            "STALP": STATES[i % len(STATES)] if i < len(STATES) * 2 else rng.choice(STATES),
            "BKCLASS": rng.choice(CHARTERS),
        })

    rows = []
    for inst in institutions:
        # Total assets in $ thousands: log-uniform between ~$150M and ~$60B.
        assets = 10 ** rng.uniform(5.2, 7.8)
        nim = rng.uniform(0.025, 0.042)
        dep_share = rng.uniform(0.74, 0.90)
        loan_share = rng.uniform(0.52, 0.76)
        efficiency = rng.uniform(0.48, 0.78)
        tier1_share = rng.uniform(0.085, 0.125)
        npl = rng.uniform(0.003, 0.025)
        for q in QUARTER_ENDS:
            assets *= 1 + rng.gauss(0.012, 0.025)
            dep_share = min(0.93, max(0.65, dep_share + rng.gauss(0, 0.012)))
            npl = min(0.06, max(0.001, npl * (1 + rng.gauss(0.02, 0.12))))
            earning = assets * rng.uniform(0.88, 0.94)
            nii = earning * (nim + rng.gauss(0, 0.0012)) / 4
            nonii = nii * rng.uniform(0.15, 0.40)
            nonix = (nii + nonii) * min(0.95, max(0.4, efficiency + rng.gauss(0, 0.03)))
            loans = assets * loan_share
            provision = loans * npl * rng.uniform(0.02, 0.10)
            netinc = (nii + nonii - nonix - provision) * 0.79
            rwa = assets * rng.uniform(0.62, 0.78)
            tier1 = rwa * tier1_share * (1 + rng.gauss(0, 0.02))
            row = {
                "CERT": inst["CERT"],
                "REPDTE": q.strftime("%Y%m%d"),
                "ASSET": _fmt(assets, rng),
                "DEP": _fmt(assets * dep_share, rng),
                "LNLSNET": _fmt(loans, rng),
                "NETINC": _fmt(netinc, rng),
                "EQ": _fmt(assets * rng.uniform(0.08, 0.12), rng),
                "NET_INT_INC": _fmt(nii, rng),
                "NONINT_INC": _fmt(nonii, rng),
                "NONINT_EXP": _fmt(nonix, rng),
                "EARNING_ASSETS": _fmt(earning, rng),
                "TIER1_CAP": _fmt(tier1, rng),
                "RWA": _fmt(rwa, rng),
                "NONCURRENT_LOANS": "" if rng.random() < 0.01 else _fmt(loans * npl, rng),
            }
            rows.append(row)

    # ~2% of rows are re-submitted (amended filings). The later copy wins in silver.
    resubmissions = []
    for row in rng.sample(rows, k=max(1, len(rows) // 50)):
        amended = dict(row)
        amended["NETINC"] = str(int(str(row["NETINC"]).replace(",", "")) + rng.randint(-500, 500))
        resubmissions.append(amended)

    _write(out_dir / "institutions.csv", INSTITUTION_FIELDS, institutions)
    _write(out_dir / "financials_batch1.csv", FINANCIAL_FIELDS, rows)
    _write(out_dir / "financials_batch2_amendments.csv", FINANCIAL_FIELDS, resubmissions)
    return {"institutions": len(institutions), "financial_rows": len(rows), "amendments": len(resubmissions)}


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    from bank_agent.config import get_settings

    print(generate(get_settings().raw_dir))
