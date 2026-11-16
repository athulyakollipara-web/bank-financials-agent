"""Pull real call report data from the FDIC BankFind Suite API into the raw layout.

The API is public and needs no key: https://api.fdic.gov/banks/docs

Usage (needs internet access)::

    python -m bank_agent.data.fetch_fdic --states MA,NY,CA --since 20240331

The FDIC exposes hundreds of call report variables. ``ASSET``, ``DEP``, ``LNLSNET``,
``NETINC`` and ``EQ`` are the well-known ones. The remaining mappings in
``FIELD_MAP`` are marked ``verify``: confirm each code against the API's
financials definition file before relying on it, and edit the map if needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

from bank_agent.data.generate import FINANCIAL_FIELDS, INSTITUTION_FIELDS

BASE_URL = "https://api.fdic.gov/banks"

# raw column in this project -> (FDIC field code, confidence)
FIELD_MAP: dict[str, tuple[str, str]] = {
    "ASSET": ("ASSET", "known"),
    "DEP": ("DEP", "known"),
    "LNLSNET": ("LNLSNET", "known"),
    "NETINC": ("NETINC", "known"),
    "EQ": ("EQ", "known"),
    "NET_INT_INC": ("NIM", "verify"),          # net interest income ($000)
    "NONINT_INC": ("NONII", "verify"),
    "NONINT_EXP": ("NONIX", "verify"),
    "EARNING_ASSETS": ("ERNAST", "verify"),
    "TIER1_CAP": ("RBCT1J", "verify"),
    "RWA": ("RWAJT", "verify"),
    "NONCURRENT_LOANS": ("NCLNLS", "verify"),
}


def _get(path: str, params: dict) -> list[dict]:
    url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed public host
        payload = json.load(resp)
    return [item["data"] for item in payload.get("data", [])]


def fetch(out_dir: Path, states: list[str], since: str, page_size: int = 5000) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_filter = " OR ".join(f"STALP:{s}" for s in states)

    institutions = _get("institutions", {
        "filters": f"({state_filter}) AND ACTIVE:1",
        "fields": "CERT,NAME,CITY,STALP,BKCLASS",
        "limit": page_size,
    })
    certs = {int(i["CERT"]) for i in institutions}

    fdic_fields = ["CERT", "REPDTE"] + [code for code, _ in FIELD_MAP.values()]
    financials: list[dict] = []
    offset = 0
    while True:
        page = _get("financials", {
            "filters": f"({state_filter}) AND REPDTE:[{since} TO *]",
            "fields": ",".join(fdic_fields),
            "limit": page_size,
            "offset": offset,
        })
        if not page:
            break
        financials.extend(page)
        offset += page_size

    rows = []
    for rec in financials:
        if int(rec.get("CERT", 0)) not in certs:
            continue
        row = {"CERT": rec["CERT"], "REPDTE": rec["REPDTE"]}
        for raw_col, (code, _) in FIELD_MAP.items():
            row[raw_col] = rec.get(code, "")
        rows.append(row)

    with (out_dir / "institutions.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INSTITUTION_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(institutions)
    with (out_dir / "financials_batch1.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINANCIAL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {"institutions": len(institutions), "financial_rows": len(rows)}


if __name__ == "__main__":
    from bank_agent.config import get_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="MA,NY,CA")
    parser.add_argument("--since", default="20240331")
    args = parser.parse_args()
    print(fetch(get_settings().raw_dir, args.states.split(","), args.since))
