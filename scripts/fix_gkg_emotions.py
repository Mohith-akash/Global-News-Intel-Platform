"""One-off fix for gkg_emotions (November 2026).

The first GCAM mapping read Roget's Thesaurus categories (c9.*) and GKG
theme counts (c18.*) as emotions, so every EMOTION_* value in the table is
wrong. The ingest now reads WordNet Affect (c15.*) per 1,000 words, and
"trust" became "confidence", the closest real dimension.

The rows can't be corrected in place (the raw GCAM field isn't stored), so
this empties the table and swaps the column. A backfill run of the ingest
workflow (GDELT_BATCHES=288, three days) then refills it with correct values.

Dry run by default; pass --apply to change the table.
usage: python scripts/fix_gkg_emotions.py [--apply]
"""

import os
import sys

import duckdb
from dotenv import load_dotenv

load_dotenv()
APPLY = "--apply" in sys.argv
db = os.getenv("GDELT_DB") or f"md:gdelt_db?motherduck_token={os.environ['MOTHERDUCK_TOKEN']}"

with duckdb.connect(db) as con:
    cols = [r[0] for r in con.execute("DESCRIBE gkg_emotions").fetchall()]
    rows = con.execute("SELECT COUNT(*) FROM gkg_emotions").fetchone()[0]
    print(f"gkg_emotions: {rows:,} rows, EMOTION_TRUST={'EMOTION_TRUST' in cols}, "
          f"EMOTION_CONFIDENCE={'EMOTION_CONFIDENCE' in cols}")
    if not APPLY:
        print("dry run: would delete all rows, drop EMOTION_TRUST, add EMOTION_CONFIDENCE")
        sys.exit(0)
    con.execute("BEGIN TRANSACTION")
    con.execute("DELETE FROM gkg_emotions")
    if "EMOTION_TRUST" in cols:
        con.execute("ALTER TABLE gkg_emotions DROP COLUMN EMOTION_TRUST")
    if "EMOTION_CONFIDENCE" not in cols:
        con.execute("ALTER TABLE gkg_emotions ADD COLUMN EMOTION_CONFIDENCE DOUBLE")
    con.execute("COMMIT")
    cols = [r[0] for r in con.execute("DESCRIBE gkg_emotions").fetchall()]
    print(f"done: {con.execute('SELECT COUNT(*) FROM gkg_emotions').fetchone()[0]} rows, columns {cols}")
