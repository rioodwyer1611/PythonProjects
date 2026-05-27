"""
lablab.ai/apps scraper
----------------------
Scrapes app listings from https://lablab.ai/apps and saves results to:
  - lablab_apps.csv
  - lablab_apps.json

Usage:
  pip install requests beautifulsoup4 lxml
  python lablab_scraper.py

Flags (edit at top of file or pass as env vars):
  MAX_PAGES   - max pages to scrape (None = all)
  DELAY       - seconds between requests (default 2.0)
  OUTPUT_DIR  - folder to save output files (default: current dir)
"""

import csv
import json
import os
import time
import random
import logging
from datetime import datetime
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://lablab.ai"
APPS_URL   = "https://lablab.ai/apps"
DELAY      = 2.0          # polite delay between requests (seconds)
JITTER     = 0.5          # random extra delay ± this value
MAX_PAGES  = None         # set to an int to limit pages, e.g. 5
OUTPUT_DIR = "."          # folder to write CSV/JSON into

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://lablab.ai/",
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── HTTP helper ───────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)


def get(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 403:
                log.error("403 Forbidden – the server is blocking this request. "
                          "Try running from a residential IP (your home machine).")
                return None
            elif resp.status_code == 429:
                wait = 30 * attempt
                log.warning(f"Rate limited. Waiting {wait}s before retry {attempt}/{retries}…")
                time.sleep(wait)
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt}/{retries})")
        except requests.RequestException as exc:
            log.warning(f"Request error on attempt {attempt}/{retries}: {exc}")

        if attempt < retries:
            time.sleep(DELAY * attempt)

    log.error(f"Failed to fetch after {retries} attempts: {url}")
    return None


def polite_sleep():
    t = DELAY + random.uniform(-JITTER, JITTER)
    time.sleep(max(t, 0.5))

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_app_card(card) -> dict:
    """Extract fields from a single app card element."""
    app = {}

    # Title
    title_el = (
        card.select_one("h2") or
        card.select_one("h3") or
        card.select_one("[class*='title']") or
        card.select_one("[class*='name']")
    )
    app["title"] = title_el.get_text(strip=True) if title_el else ""

    # Description
    desc_el = (
        card.select_one("p") or
        card.select_one("[class*='description']") or
        card.select_one("[class*='desc']")
    )
    app["description"] = desc_el.get_text(strip=True) if desc_el else ""

    # URL / detail link
    link_el = card.select_one("a[href]")
    if link_el:
        href = link_el["href"]
        app["url"] = href if href.startswith("http") else urljoin(BASE_URL, href)
    else:
        app["url"] = ""

    # Tech stack / tags (look for badge-like elements)
    tags = []
    for tag_el in card.select("[class*='tag'], [class*='badge'], [class*='tech'], [class*='label']"):
        t = tag_el.get_text(strip=True)
        if t:
            tags.append(t)
    app["tags"] = ", ".join(tags)

    # Votes / likes
    vote_el = (
        card.select_one("[class*='vote']") or
        card.select_one("[class*='like']") or
        card.select_one("[class*='upvote']")
    )
    app["votes"] = vote_el.get_text(strip=True) if vote_el else ""

    # Thumbnail image
    img_el = card.select_one("img")
    app["image_url"] = img_el.get("src", "") if img_el else ""

    return app


def parse_app_detail(url: str) -> dict:
    """
    Optionally fetch an app's detail page for richer data.
    Returns a dict of extra fields to merge into the card data.
    """
    extra = {}
    soup = get(url)
    if not soup:
        return extra

    # Authors / team
    author_el = soup.select_one("[class*='author'], [class*='team'], [class*='creator']")
    extra["author"] = author_el.get_text(strip=True) if author_el else ""

    # Full description (detail page often has more)
    full_desc = soup.select_one("article, [class*='content'], [class*='body']")
    if full_desc:
        extra["full_description"] = full_desc.get_text(" ", strip=True)[:2000]

    # GitHub / demo links
    links = {}
    for a in soup.select("a[href]"):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if "github.com" in href:
            links["github"] = href
        elif "demo" in text or "live" in text:
            links["demo"] = href
    extra.update(links)

    return extra


def find_next_page(soup, current_url: str) -> str | None:
    """Try to find a 'next page' link from pagination elements."""
    # Common pagination patterns
    for selector in [
        "a[aria-label='Next']",
        "a[rel='next']",
        "a:contains('Next')",
        "[class*='pagination'] a:last-child",
        "[class*='next'] a",
        "a[class*='next']",
    ]:
        el = soup.select_one(selector)
        if el and el.get("href"):
            href = el["href"]
            return href if href.startswith("http") else urljoin(BASE_URL, href)

    # Fallback: look for numbered page links and find current + 1
    page_links = soup.select("[class*='pagination'] a[href], [class*='pager'] a[href]")
    for i, el in enumerate(page_links):
        if el.get("aria-current") or "active" in el.get("class", []):
            # return the next sibling link if it exists
            if i + 1 < len(page_links):
                href = page_links[i + 1]["href"]
                return href if href.startswith("http") else urljoin(BASE_URL, href)

    return None


def find_app_cards(soup) -> list:
    """Find all app card elements on the page."""
    # Try progressively broader selectors
    for selector in [
        "[class*='AppCard']",
        "[class*='app-card']",
        "[class*='appCard']",
        "article",
        "[class*='card']",
        "li[class*='app']",
    ]:
        cards = soup.select(selector)
        # Filter out cards that are clearly nav/footer noise (too little text)
        cards = [c for c in cards if len(c.get_text(strip=True)) > 30]
        if cards:
            log.info(f"Found {len(cards)} cards with selector: {selector!r}")
            return cards

    log.warning("Could not identify app card elements – dumping all <article> tags")
    return soup.select("article") or []

# ── Save helpers ──────────────────────────────────────────────────────────────

def save_csv(apps: list[dict], path: str):
    if not apps:
        return
    fieldnames = list(apps[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(apps)
    log.info(f"Saved CSV  → {path}")


def save_json(apps: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(apps, f, ensure_ascii=False, indent=2)
    log.info(f"Saved JSON → {path}")

# ── Main scrape loop ──────────────────────────────────────────────────────────

def scrape(
    fetch_detail_pages: bool = False,
    max_pages: int | None = MAX_PAGES,
) -> list[dict]:
    """
    Scrape lablab.ai/apps.

    Args:
        fetch_detail_pages: If True, visit each app's page for richer data
                            (slower – one extra request per app).
        max_pages:          Stop after this many listing pages (None = all).

    Returns:
        List of app dicts.
    """
    all_apps: list[dict] = []
    seen_urls: set[str] = set()
    url = APPS_URL
    page_num = 0

    log.info(f"Starting scrape of {APPS_URL}")
    log.info(f"Settings: delay={DELAY}s, max_pages={max_pages or 'unlimited'}, "
             f"detail_pages={fetch_detail_pages}")

    while url:
        page_num += 1
        if max_pages and page_num > max_pages:
            log.info(f"Reached page limit ({max_pages}). Stopping.")
            break

        log.info(f"── Page {page_num}: {url}")
        soup = get(url)

        if soup is None:
            log.error("Could not fetch page. Stopping.")
            break

        cards = find_app_cards(soup)
        if not cards:
            log.warning("No app cards found on this page. The site structure may have changed.")
            # Dump raw HTML snippet for debugging
            with open("debug_page.html", "w", encoding="utf-8") as f:
                f.write(soup.prettify()[:50_000])
            log.info("Saved first 50k chars of page HTML to debug_page.html for inspection.")
            break

        page_apps = []
        for card in cards:
            app = parse_app_card(card)

            # Skip duplicates
            if app["url"] and app["url"] in seen_urls:
                continue
            if app["url"]:
                seen_urls.add(app["url"])

            app["scraped_at"] = datetime.utcnow().isoformat()

            # Optionally enrich from detail page
            if fetch_detail_pages and app["url"]:
                polite_sleep()
                extra = parse_app_detail(app["url"])
                app.update(extra)

            page_apps.append(app)

        log.info(f"   Collected {len(page_apps)} apps (total so far: {len(all_apps) + len(page_apps)})")
        all_apps.extend(page_apps)

        # Pagination
        next_url = find_next_page(soup, url)
        if next_url and next_url != url:
            polite_sleep()
            url = next_url
        else:
            log.info("No next page found – scrape complete.")
            break

    return all_apps


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape lablab.ai/apps")
    parser.add_argument(
        "--detail", action="store_true",
        help="Also fetch each app's detail page for richer data (slower)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Max listing pages to scrape (default: all)"
    )
    parser.add_argument(
        "--delay", type=float, default=DELAY,
        help=f"Seconds between requests (default: {DELAY})"
    )
    parser.add_argument(
        "--out", type=str, default=OUTPUT_DIR,
        help="Output directory for CSV/JSON files"
    )
    args = parser.parse_args()

    DELAY = args.delay
    os.makedirs(args.out, exist_ok=True)

    apps = scrape(fetch_detail_pages=args.detail, max_pages=args.max_pages)

    if apps:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_csv(apps,  os.path.join(args.out, f"lablab_apps_{ts}.csv"))
        save_json(apps, os.path.join(args.out, f"lablab_apps_{ts}.json"))
        log.info(f"Done. {len(apps)} apps saved.")
    else:
        log.warning("No apps were scraped. Check the debug output above.")