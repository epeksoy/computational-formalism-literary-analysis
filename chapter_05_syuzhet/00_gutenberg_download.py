#!/usr/bin/env python3
"""
Download Project Gutenberg texts by ID and strip the PG header/footer.

Usage
-----
    python 00_gutenberg_download.py ids.txt
    python 00_gutenberg_download.py ids.txt --out corpus --delay 1.0 --workers 2
    python 00_gutenberg_download.py ids.txt --keep-raw        # also save untrimmed text
    python 00_gutenberg_download.py ids.txt --overwrite       # re-download existing files

Input file: one numeric Gutenberg ID per line (blank lines and lines starting
with '#' are ignored).

Output: corpus/<id>.txt, UTF-8, header/footer removed.
Failures are listed in corpus/_failed.txt.

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (compatible; corpus-builder/1.0; "
    "academic text-analysis research; contact: you@example.org)"
)

# Tried in order. The cache/epub path is the most reliable for modern IDs;
# the -0 / -8 / plain variants under /files/ cover older postings.
URL_TEMPLATES = (
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}-8.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
)

# Encodings tried in order when the server gives no usable charset.
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def fetch(url: str, timeout: float = 60.0) -> str | None:
    """Fetch one URL and decode it. Returns None on 404."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset()

    if charset:
        try:
            return raw.decode(charset, errors="replace")
        except LookupError:
            pass
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def download(book_id: str, retries: int = 3, delay: float = 1.0) -> str:
    """Try each URL template; retry on transient errors. Raises on failure."""
    last_error: Exception | None = None
    for url in (t.format(id=book_id) for t in URL_TEMPLATES):
        for attempt in range(retries):
            try:
                text = fetch(url)
                if text and text.strip():
                    return text
                break
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 404:
                    break  # this variant does not exist; try the next template
                if e.code in (403, 429, 503):
                    # Gutenberg throttles aggressive clients: back off hard.
                    time.sleep(delay * (5 ** (attempt + 1)) + random.random())
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = e
                time.sleep(delay * (2 ** attempt) + random.random())
    raise RuntimeError(f"could not download {book_id}: {last_error}")


# --------------------------------------------------------------------------
# Header / footer removal
# --------------------------------------------------------------------------

START_PATTERNS = (
    # Modern:  *** START OF THE PROJECT GUTENBERG EBOOK TITLE ***
    re.compile(
        r"^[^\S\n]*\*{2,}[^\S\n]*START OF (?:TH(?:E|IS)[^\S\n]+)?"
        r"PROJECT GUTENBERG.*?$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Pre-2004 boilerplate terminator.
    re.compile(
        r"^.*\*END\*[^\S\n]*THE SMALL PRINT.*?$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^[^\S\n]*\*{2,}[^\S\n]*START OF TH(?:E|IS) PROJECT GUTENBERG "
        r"(?:E?TEXT|E?BOOK).*?$",
        re.IGNORECASE | re.MULTILINE,
    ),
)

END_PATTERNS = (
    re.compile(
        r"^[^\S\n]*\*{2,}[^\S\n]*END OF (?:TH(?:E|IS)[^\S\n]+)?"
        r"PROJECT GUTENBERG.*?$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^[^\S\n]*End of (?:the |The )?Project Gutenberg(?:'s)?\b.*?$",
        re.MULTILINE,
    ),
    re.compile(
        r"^[^\S\n]*End of th(?:e|is) Project Gutenberg (?:E?Text|E?Book).*?$",
        re.IGNORECASE | re.MULTILINE,
    ),
)

# Lines that survive the start marker but are not part of the work.
CREDIT_LINE = re.compile(
    r"^[^\S\n]*(?:Produced by|Transcribed (?:from|by)|E-?text prepared by|"
    r"This e?-?book was produced by|Updated editions will|Credits:|"
    r"Release [Dd]ate:|Language:|Character set encoding:|Title:|Author:|"
    r"Translator:|Editor:|Illustrator:|Posting Date:|\[?Illustration)",
)


def strip_gutenberg(text: str) -> str:
    """Remove PG front matter and back matter; return the work itself."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    # --- front matter: cut everything up to and including the last start
    # marker that occurs in the first half of the file.
    start = 0
    for pattern in START_PATTERNS:
        for m in pattern.finditer(text):
            if m.end() > len(text) // 2:
                break
            start = max(start, m.end())
        if start:
            break

    # --- back matter: cut from the first end marker after the start.
    end = len(text)
    for pattern in END_PATTERNS:
        m = pattern.search(text, start)
        if m:
            end = min(end, m.start())

    body = text[start:end]

    # Drop residual credit/metadata lines and blank space at the top.
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or CREDIT_LINE.match(line):
            i += 1
            continue
        break
    body = "\n".join(lines[i:])

    # Collapse runs of more than two blank lines; normalise the tail.
    body = re.sub(r"\n{4,}", "\n\n\n", body)
    return body.strip() + "\n"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.isdigit():
            print(f"  skipping non-numeric line: {line!r}", file=sys.stderr)
            continue
        if line not in seen:
            seen.add(line)
            ids.append(line)
    return ids


def process(book_id: str, out_dir: Path, args, lock: Lock) -> tuple[str, str]:
    """Returns (book_id, status) where status is ok / skipped / an error."""
    target = out_dir / f"{book_id}.txt"
    if target.exists() and target.stat().st_size > 0 and not args.overwrite:
        return book_id, "skipped"

    try:
        raw = download(book_id, retries=args.retries, delay=args.delay)
    except Exception as e:  # noqa: BLE001 - report and continue
        return book_id, f"FAILED ({e})"

    if args.keep_raw:
        raw_dir = out_dir / "_raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / f"{book_id}.txt").write_text(raw, encoding="utf-8")

    body = strip_gutenberg(raw)
    if len(body) < args.min_chars:
        return book_id, f"FAILED (only {len(body)} chars after stripping)"

    target.write_text(body, encoding="utf-8")

    with lock:
        time.sleep(args.delay)  # be polite to the server
    return book_id, "ok"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ids", type=Path, help="file with one Gutenberg ID per line")
    p.add_argument("--out", type=Path, default=Path("corpus"),
                   help="output folder (default: corpus)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds to pause between downloads (default: 1.0)")
    p.add_argument("--workers", type=int, default=2,
                   help="parallel downloads; keep low (default: 2)")
    p.add_argument("--retries", type=int, default=3,
                   help="retries per URL variant (default: 3)")
    p.add_argument("--min-chars", type=int, default=1000,
                   help="reject stripped texts shorter than this (default: 1000)")
    p.add_argument("--keep-raw", action="store_true",
                   help="also store the untrimmed file in corpus/_raw/")
    p.add_argument("--overwrite", action="store_true",
                   help="re-download IDs that already have a file")
    args = p.parse_args()

    if not args.ids.exists():
        print(f"no such file: {args.ids}", file=sys.stderr)
        return 1

    ids = read_ids(args.ids)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} IDs -> {args.out.resolve()}")

    lock = Lock()
    ok = skipped = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process, i, args.out, args, lock): i for i in ids}
        for n, fut in enumerate(as_completed(futures), start=1):
            book_id, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append(f"{book_id}\t{status}")
            print(f"[{n}/{len(ids)}] {book_id}: {status}", flush=True)

    if failures:
        (args.out / "_failed.txt").write_text("\n".join(sorted(failures)) + "\n",
                                              encoding="utf-8")
    print(f"\ndone: {ok} downloaded, {skipped} already present, "
          f"{len(failures)} failed")
    if failures:
        print(f"failed IDs listed in {args.out / '_failed.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
