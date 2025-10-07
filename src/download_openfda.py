#!/usr/bin/env python3
"""
download_from_saved_openfda_html.py

Parses a locally saved OpenFDA 'Downloads' HTML fragment and downloads all dataset files it references,
replicating the remote URL folder structure locally and resuming safely.

Requirements:
    pip install requests beautifulsoup4 tqdm
"""

import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIG ---
FILE_PATH = "openfda_downloads/2025-10-02.faers.download.html"  # your saved HTML fragment
BASE_URL  = "https://open.fda.gov/data/downloads/"              # for resolving relative hrefs
OUT_DIR   = "openfda_downloads/files"
ALLOWED_EXT = {".zip", ".gz", ".json", ".csv", ".ndjson", ".xml"}
DOMAIN_ALLOWLIST = {"download.open.fda.gov", "api.fda.gov", "open.fda.gov"}
MAX_WORKERS = 4
CHUNK_SIZE = 1024 * 1024  # 1 MB
TIMEOUT = 60
RETRIES = 3

# Optional: restrict to certain collections, e.g., ["faers", "drug/event"]
FILTER_SUBSTRINGS = []  # e.g., ["faers"]

# --- Helpers ---

def read_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def has_allowed_ext(url):
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in ALLOWED_EXT)

def allowed_domain(url):
    host = urlparse(url).hostname or ""
    return any(host.endswith(d) for d in DOMAIN_ALLOWLIST)

def passes_filters(url):
    if FILTER_SUBSTRINGS:
        u = url.lower()
        return any(s in u for s in FILTER_SUBSTRINGS)
    return True

def dedupe_preserve_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_urls_from_html(html, base=BASE_URL):
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    # 1) Plain anchors
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        urls.append(urljoin(base, href))

    # 2) onclick attributes
    for tag in soup.find_all(True):
        onclick = tag.get("onclick")
        if onclick:
            urls += re.findall(r"https?://[^\s'\"<>]+", onclick)

    # 3) Raw text + <script> contents
    blobs = [soup.get_text("\n", strip=False)]
    for script in soup.find_all("script"):
        if script.string:
            blobs.append(script.string)
    urls += re.findall(r"https?://[^\s'\"<>]+", "\n".join(blobs))

    # Normalize, filter, dedupe
    urls = [
        u for u in (u.strip() for u in urls)
        if allowed_domain(u) and has_allowed_ext(u) and passes_filters(u)
    ]
    return dedupe_preserve_order(urls)

def url_to_local_path(url, root=OUT_DIR):
    """
    Replicate full URL path under OUT_DIR, including host.
    e.g. https://download.open.fda.gov/drug/event/2025q1/file.gz ->
         OUT_DIR/download.open.fda.gov/drug/event/2025q1/file.gz
    """
    pu = urlparse(url)
    rel = os.path.join(pu.hostname or "unknown-host", pu.path.lstrip("/"))
    # drop query string semantics in filename level, but keep path structure
    base, name = os.path.split(rel)
    name = name.split("?")[0]
    rel = os.path.join(base, name)
    full = os.path.join(root, rel)
    return full

def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def head_info(url):
    """
    Return (content_length:int|None, accept_ranges:bool)
    """
    try:
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
        if 200 <= r.status_code < 400:
            cl = r.headers.get("Content-Length")
            length = int(cl) if cl and cl.isdigit() else None
            ar = r.headers.get("Accept-Ranges", "").lower()
            accept_ranges = "bytes" in ar
            return length, accept_ranges
    except requests.RequestException:
        pass
    return None, False

def download_with_resume(url, final_path):
    """
    Safe, resumable downloader:
      - uses .part temp file
      - resumes if .part exists and server supports Range
      - validates full-size completion against Content-Length when available
    """
    ensure_parent_dir(final_path)
    tmp_path = final_path + ".part"

    remote_len, accept_ranges = head_info(url)

    # Case 1: final file already complete
    if os.path.exists(final_path) and remote_len is not None:
        if os.path.getsize(final_path) == remote_len:
            return "exists"

    # Case 2: figure out resume offset (if .part exists and server supports it)
    resume_from = 0
    if os.path.exists(tmp_path):
        part_size = os.path.getsize(tmp_path)
        if remote_len is not None and part_size > remote_len:
            # local partial is larger than remote -> nuke and restart
            os.remove(tmp_path)
        elif accept_ranges and part_size > 0:
            resume_from = part_size
        else:
            # cannot resume safely; restart
            os.remove(tmp_path)

    headers = {}
    mode = "wb"
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    # Stream download
    with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as r:
        # Handle 416 (range not satisfiable) by falling back to full restart
        if r.status_code == 416:
            # server thinks we already have it all; verify and rename
            if os.path.exists(tmp_path) and remote_len is not None and os.path.getsize(tmp_path) == remote_len:
                os.replace(tmp_path, final_path)
                return "ok"
            # else restart
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return download_with_resume(url, final_path)

        r.raise_for_status()
        # total to show in progress bar
        total_from_headers = r.headers.get("Content-Length")
        try:
            total = int(total_from_headers) if total_from_headers and total_from_headers.isdigit() else None
        except ValueError:
            total = None

        # When resuming, total should be remaining bytes; we adjust bar initial with resume_from for clarity
        initial = resume_from
        total_display = (resume_from + total) if (total is not None) else None

        with open(tmp_path, mode) as f, tqdm(
            total=total_display,
            initial=initial,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(final_path),
            leave=True
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Verify complete if we know size
    if remote_len is not None and os.path.getsize(tmp_path) != remote_len:
        # Incomplete; leave .part for future resume
        return "partial"

    # Promote to final
    os.replace(tmp_path, final_path)
    return "ok"

def download_one(url):
    final_path = url_to_local_path(url, OUT_DIR)
    for attempt in range(1, RETRIES + 1):
        try:
            status = download_with_resume(url, final_path)
            return url, final_path, status
        except Exception as e:
            # On exceptions, keep .part; we'll resume next attempt
            if attempt < RETRIES:
                wait = 2 ** (attempt - 1)
                print(f"[{final_path}] attempt {attempt} failed: {e!r}. retrying in {wait}s...")
                time.sleep(wait)
            else:
                return url, final_path, f"error: {e!r}"

def main():
    html = read_html(FILE_PATH)
    urls = extract_urls_from_html(html, base=BASE_URL)
    if not urls:
        print("No candidate file URLs found. Tip: check ALLOWED_EXT, DOMAIN_ALLOWLIST, or FILTER_SUBSTRINGS.")
        return

    print(f"Found {len(urls)} file(s). Downloading under: {OUT_DIR}")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(download_one, u): u for u in urls}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = (u, url_to_local_path(u, OUT_DIR), f"error: {e!r}")
            results.append(res)
            print(f"{res[0]} -> {res[2]} -> {res[1]}")

    ok = [r for r in results if r[2] in ("ok", "exists")]
    partial = [r for r in results if r[2] == "partial"]
    errs = [r for r in results if r[2].startswith("error")]
    print(f"\nSummary: OK/EXISTS={len(ok)}; PARTIAL={len(partial)}; ERRORS={len(errs)}")
    if partial:
        print("Partial (will resume next run):")
        for u, p, _ in partial:
            print(" -", p)
    if errs:
        print("Errors:")
        for u, p, msg in errs:
            print(" -", u, "->", msg)

if __name__ == "__main__":
    main()
