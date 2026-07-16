"""
01_extract_5repos.py
Month-stratified extraction of the 5 high-corporate-pressure /
critical-infrastructure repositories. Resumable: if interrupted, running
the same command again continues where it left off, without duplicating data.

Repos: facebook/react, tensorflow/tensorflow, microsoft/vscode,
       torvalds/linux, bitcoin/bitcoin
Window: 2025-01-01 to 2026-07-14, stratified sampling by calendar month.

Note: internal DataFrame/CSV column names are in English, matching the
released data/raw/extraction_trace_final.csv file.
"""
import requests
import pandas as pd
import time
import os
import re
import random
from datetime import datetime

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', 'YOUR_TOKEN_HERE')

REPOSITORIES = [
    'facebook/react',
    'tensorflow/tensorflow',
    'microsoft/vscode',
    'torvalds/linux',
    'bitcoin/bitcoin',
]

WINDOW_START = datetime(2025, 1, 1)
WINDOW_END = datetime(2026, 7, 14)

TARGET_PER_REPO_MONTH = 200
CANDIDATE_CAP_PER_MONTH = 800
OUTPUT_FILE = 'data/raw/github_raw_v3.parquet'
TRACE_FILE = 'data/raw/extraction_trace_v3.csv'

HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

BOT_PATTERNS = re.compile(
    r'(bot|gardener|copilot|\[bot\]|ci|actions|dependabot|jenkins|travis)',
    re.IGNORECASE
)
JUNK_PATTERN = re.compile(r'[~`\*\#\}\{\+\|\&\^\_\$]{2,}')


def check_rate_limit(response):
    if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
        if int(response.headers['X-RateLimit-Remaining']) == 0:
            reset_time = int(response.headers['X-RateLimit-Reset'])
            sleep_time = max(0, reset_time - int(time.time())) + 10
            print(f"\n[!] API rate limit reached. Sleeping {sleep_time / 60:.1f} min...")
            time.sleep(sleep_time)
            return True
    return False


def is_bot(login, name, email):
    candidates = [login or '', name or '', email or '']
    return any(BOT_PATTERNS.search(c) for c in candidates)


def is_junk_identity(login, name, email):
    for c in [login, name, email]:
        if c and JUNK_PATTERN.search(c):
            return True
    return False


def month_ranges(start, end):
    cur = datetime(start.year, start.month, 1)
    while cur < end:
        nxt = datetime(cur.year + 1, 1, 1) if cur.month == 12 else datetime(cur.year, cur.month + 1, 1)
        yield (max(cur, start), min(nxt, end))
        cur = nxt


def extract_commits():
    os.makedirs('data/raw', exist_ok=True)

    # --- Safe resume: skip months already fully processed ---
    if os.path.exists(TRACE_FILE):
        trace_rows = pd.read_csv(TRACE_FILE).to_dict('records')
        done_months = {(r['repo'], r['month']) for r in trace_rows}
    else:
        trace_rows = []
        done_months = set()

    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_parquet(OUTPUT_FILE)
        existing_df['ym'] = pd.to_datetime(existing_df['date'].str[:19]).dt.strftime('%Y-%m')
        mask_complete = existing_df.apply(lambda r: (r['repo'], r['ym']) in done_months, axis=1)
        kept_df = existing_df[mask_complete].drop(columns=['ym'])
        all_data = kept_df.to_dict('records')
        print(f"Resuming: {len(all_data)} valid commits from completed months.")
    else:
        all_data = []

    for repo in REPOSITORIES:
        print(f"\nRepository: {repo}")

        for m_start, m_end in month_ranges(WINDOW_START, WINDOW_END):
            mes_str = m_start.strftime('%Y-%m')
            if (repo, mes_str) in done_months:
                print(f"   {mes_str}: already completed, skipping.")
                continue

            since = m_start.strftime('%Y-%m-%dT00:00:00Z')
            until = m_end.strftime('%Y-%m-%dT00:00:00Z')

            candidates = []
            bots_excluded = 0
            junk_excluded = 0
            page = 1

            # --- Step 1: list candidates (cheap, list-only API calls) ---
            while len(candidates) < CANDIDATE_CAP_PER_MONTH:
                url_list = (
                    f'https://api.github.com/repos/{repo}/commits'
                    f'?per_page=100&page={page}&since={since}&until={until}'
                )
                res_list = requests.get(url_list, headers=HEADERS)
                if check_rate_limit(res_list):
                    continue
                if res_list.status_code != 200:
                    print(f"   Error {res_list.status_code}: {res_list.text[:150]}")
                    break

                commits = res_list.json()
                if not commits:
                    break

                for item in commits:
                    gh_login = item['author']['login'] if item.get('author') else None
                    raw_name = item['commit']['author'].get('name')
                    raw_email = item['commit']['author'].get('email')

                    if is_bot(gh_login, raw_name, raw_email):
                        bots_excluded += 1
                        continue
                    if is_junk_identity(gh_login, raw_name, raw_email):
                        junk_excluded += 1
                        continue

                    candidates.append({
                        'sha': item['sha'],
                        'author_id': gh_login or raw_email or raw_name or 'unknown',
                        'author_name_raw': raw_name,
                        'date': item['commit']['author']['date'],
                        'commit_message': item['commit']['message'],
                    })

                page += 1
                if len(commits) < 100:
                    break

            # --- Step 2: filter by real AUTHOR date within the global window ---
            candidates = [
                c for c in candidates
                if WINDOW_START <= datetime.strptime(c['date'][:19], '%Y-%m-%dT%H:%M:%S') <= WINDOW_END
            ]

            # --- Step 3: random sample of the month ---
            sample = random.sample(candidates, min(TARGET_PER_REPO_MONTH, len(candidates)))

            # --- Step 4: fetch detail (added/deleted lines) only for the sample ---
            included = 0
            for c in sample:
                url_detail = f"https://api.github.com/repos/{repo}/commits/{c['sha']}"
                res_detail = requests.get(url_detail, headers=HEADERS)
                if check_rate_limit(res_detail):
                    res_detail = requests.get(url_detail, headers=HEADERS)
                if res_detail.status_code != 200:
                    continue

                stats = res_detail.json().get('stats', {})
                all_data.append({
                    'repo': repo,
                    'sha': c['sha'],
                    'author_id': c['author_id'],
                    'author_name_raw': c['author_name_raw'],
                    'date': c['date'],
                    'commit_message': c['commit_message'],
                    'lines_added': stats.get('additions', 0),
                    'lines_deleted': stats.get('deletions', 0),
                })
                included += 1

            trace_rows.append({
                'repo': repo, 'month': mes_str,
                'candidates_seen': len(candidates) + bots_excluded + junk_excluded,
                'excluded_bots': bots_excluded, 'excluded_junk': junk_excluded,
                'valid_candidates': len(candidates), 'sampled': len(sample),
                'included_final': included,
            })
            print(f"   {mes_str}: {included} included "
                  f"(valid: {len(candidates)}, bots: {bots_excluded}, junk: {junk_excluded})")

            # Incremental save per month: a rate-limit cutoff doesn't lose progress
            pd.DataFrame(trace_rows).to_csv(TRACE_FILE, index=False)
            pd.DataFrame(all_data).to_parquet(OUTPUT_FILE, index=False)

    print(f"\nDone: {len(all_data)} commits in {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_commits()
