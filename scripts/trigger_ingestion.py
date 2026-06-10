"""
External cron trigger for GDELT 15-min ingestion.

GitHub Actions' built-in scheduler is unreliable for sub-hourly crons —
runs are delayed 1-4 hours during peak load. This script is called by
cron-job.org every 15 minutes to trigger workflow_dispatch reliably.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONE-TIME SETUP  (~8 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Create a GitHub PAT
  https://github.com/settings/tokens/new
  Name:   gdelt-cron-trigger
  Expiry: No expiration  (or 1 year — set a calendar reminder)
  Scope:  ✅ workflow   (under repo section)
  → Copy the token — you'll only see it once

Step 2 — Add it as a repo secret
  https://github.com/Mohith-akash/Global-News-Intel-Platform/settings/secrets/actions/new
  Name:   DISPATCH_TOKEN
  Value:  <paste PAT from step 1>

Step 3 — Create a cron-job.org account
  https://cron-job.org  (free, no credit card)

Step 4 — Create a new cron job with these exact settings:
  Title:    GDELT 15-min trigger
  URL:      https://api.github.com/repos/Mohith-akash/Global-News-Intel-Platform/actions/workflows/gdelt_ingest.yml/dispatches
  Interval: Every 15 minutes
  Method:   POST
  Headers:
    Accept: application/vnd.github+json
    Authorization: Bearer <your PAT from step 1>
    X-GitHub-Api-Version: 2022-11-28
  Body (JSON):
    {"ref": "master"}

  Expected response: 204 No Content  (GitHub returns no body on success)
  Enable notifications on failure: ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TESTING  (verify it works before relying on cron-job.org)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run this script locally with your PAT to confirm dispatch works:
  python scripts/trigger_ingestion.py --token ghp_xxxxxxxxxxxx

Then check:
  https://github.com/Mohith-akash/Global-News-Intel-Platform/actions

A new run titled "15-Min GDELT Refresh (Polars)" should appear within
a few seconds. If it does, the cron-job.org setup will work identically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE AFTER SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cron-job.org (every 15 min)
       │
       └─ POST /repos/.../dispatches  ──►  GitHub Actions runs immediately
                                           (workflow_dispatch is not throttled)

  GitHub cron (hourly, 0 * * * *)
       │
       └─ Fallback if cron-job.org is down or misconfigured
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import json

REPO = "Mohith-akash/Global-News-Intel-Platform"
WORKFLOW = "gdelt_ingest.yml"
REF = "master"


def dispatch(token: str) -> None:
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW}/dispatches"
    payload = json.dumps({"ref": REF}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            # 204 No Content = success
            print(f"✅ Dispatched — HTTP {resp.status}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"❌ HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger GDELT ingestion workflow")
    parser.add_argument("--token", default=os.getenv("DISPATCH_TOKEN"),
                        help="GitHub PAT with workflow scope (or set DISPATCH_TOKEN env var)")
    args = parser.parse_args()

    if not args.token:
        print("❌ No token. Pass --token ghp_xxx or set DISPATCH_TOKEN env var.", file=sys.stderr)
        sys.exit(1)

    dispatch(args.token)
