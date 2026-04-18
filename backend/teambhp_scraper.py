"""
TeamBHP Forum Thread Scraper
----------------------------
Uses nodriver (real Chrome, no headless flag) to bypass Cloudflare Bot Management,
then BeautifulSoup to extract posts.  Each post is returned as a discrete dict so
the chunker can treat it as one chunk.

Why nodriver?
  Cloudflare Bot Management blocks headless Chrome by fingerprinting canvas, fonts,
  GPU, and automation flags.  nodriver starts an unmodified real Chrome process
  (no --headless, no CDP patches) so CF cannot distinguish it from a human user.
"""

import re
import asyncio
import random
import hashlib
from typing import List, Dict, Optional
from urllib.parse import urljoin


# ── URL helpers ────────────────────────────────────────────────────────────────

def is_teambhp_url(url: str) -> bool:
    return "team-bhp.com" in url.lower()


def extract_thread_id(url: str) -> str:
    """
    Pull the numeric thread-ID from a TeamBHP URL.
    .../305630-skoda-kylaq-...html  →  "305630"
    Falls back to an MD5 prefix if no number is found.
    """
    match = re.search(r"/(\d{4,})-[^/]", url)
    if match:
        return match.group(1)
    return hashlib.md5(url.encode()).hexdigest()[:10]


# ── Cloudflare helpers ─────────────────────────────────────────────────────────

def _is_cloudflare_block(html: str, title: str) -> bool:
    """Return True when the page is a Cloudflare challenge/block page."""
    cf_markers = [
        "Attention Required! | Cloudflare",
        "Just a moment...",
        "cf-error-details",
        "Please enable cookies",
        "You are unable to access",
    ]
    return any(m in html or m in title for m in cf_markers)


# ── HTML parsing ───────────────────────────────────────────────────────────────

def _parse_posts_from_html(html: str) -> List[Dict]:
    """
    Extract posts from a TeamBHP (vBulletin) page.

    vBulletin uses several divs whose IDs all start with "post_":
      - post_NNNN          ← the actual post container  (what we want)
      - post_message_NNNN  ← post body text             (nested inside post_NNNN)
      - post_thanks_row_NNNN ← "Thanks" userlist        (what was wrongly captured)
      - post_thanks_NNNN, post_userinfo_NNNN, …          (other vB helpers)

    Fix: select ONLY elements whose id is literally "post_" + pure digits.
    Then use the numeric part to look up the body (post_message_NNNN) and
    author (postmenu_NNNN) elements directly by ID.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("beautifulsoup4 is required: pip install beautifulsoup4 lxml")

    soup = BeautifulSoup(html, "html.parser")
    posts: List[Dict] = []

    # ── 1. vBulletin 3.x post containers: id="post12345" (NO underscore) ────────
    #    All other vB elements that start with "post" use underscores:
    #      post_message_12345, post_thanks_row_12345, postmenu_12345 …
    #    Filtering to ^post\d+$ therefore isolates ONLY real post containers.
    all_candidates = soup.select(
        "table[id^='post'], div[id^='post'], tr[id^='post'], li[id^='post']"
    )
    containers = [
        el for el in all_candidates
        if re.match(r"^post\d+$", el.get("id", ""))   # "post12345" — no underscore
    ]

    # ── 2. XenForo / generic fallback ─────────────────────────────────────────
    if not containers:
        containers = (
            soup.select("article.message")
            or soup.select("div.post")
            or soup.select("li[id^='post-']")
        )

    for container in containers:
        post_id = container.get("id", "")
        # Extract the trailing digits regardless of format
        # "post12345" → "12345",  "post_12345" → "12345"
        num_match = re.search(r"\d+$", post_id)
        num_id = num_match.group() if num_match else ""

        # ── Author ─────────────────────────────────────────────────────────────
        # vBulletin 3.x: <div id="postmenu_12345"><a …>Username</a></div>
        author = ""
        if num_id:
            postmenu = soup.find(id=f"postmenu_{num_id}")
            if postmenu:
                a = postmenu.find("a")
                if a:
                    author = a.get_text(strip=True)

        if not author:
            for sel in [
                "a.bigusername", "span.largetext a", ".username a",
                "a[class*='username']", "span.username", ".post-author",
                "h4.message-name a",
            ]:
                el = container.select_one(sel)
                if el and el.get_text(strip=True):
                    author = el.get_text(strip=True)
                    break

        # ── Date ───────────────────────────────────────────────────────────────
        # vBulletin 3.x stores the date in the first td.thead inside the post
        post_date = ""
        for sel in [
            "td.thead", "span.date", "time",
            "span[class*='date']", "abbr.DateTime", "span.DateTime", ".date",
        ]:
            el = container.select_one(sel)
            if el:
                post_date = el.get("title") or el.get_text(strip=True)
                if post_date:
                    break

        # ── Post text ──────────────────────────────────────────────────────────
        # Primary: id="post_message_NNNN" — the dedicated vBulletin message div.
        # Look it up by ID so we never accidentally grab a sibling thanks row.
        text = ""
        if num_id:
            msg_el = soup.find(id=f"post_message_{num_id}")
            if msg_el:
                # Remove any nested thanks / userlist noise before extracting text
                for junk in msg_el.select(
                    "[id*='thanks'], [class*='thanks'], [class*='userlist'], td.alt2"
                ):
                    junk.decompose()
                text = msg_el.get_text(separator="\n", strip=True)

        # Fallback for non-vBulletin or future layout changes
        if not text:
            for sel in [
                "blockquote.postcontent",
                "div.post-content", "div.postcontent", "div.post-message",
                "div[class*='post-body']", "div.bbWrapper", "div.message-body",
            ]:
                el = container.select_one(sel)
                if el and el.get_text(strip=True):
                    text = el.get_text(separator="\n", strip=True)
                    break

        if text and len(text.strip()) > 30:
            posts.append({
                "post_id": post_id,
                "author":  author or "Unknown",
                "date":    post_date,
                "text":    text.strip(),
            })

    return posts


def _get_thread_title(html: str, tab_title: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for sel in ["h1.thread-title", "h2.threadtitle", "#pagetitle h1", "h1"]:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                return el.get_text(strip=True)
    except Exception:
        pass
    return tab_title.split(" - ")[0].strip() if " - " in tab_title else tab_title


def _get_next_page_url(html: str, current_url: str) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        soup = BeautifulSoup(html, "html.parser")

        el = soup.find("a", rel="next")
        if el and el.get("href"):
            href = el["href"]
            return href if href.startswith("http") else urljoin(current_url, href)

        NEXT_TEXTS = {"next", "next »", "›", "»", "next page", ">"}
        for a in soup.select("a[href]"):
            if a.get_text(strip=True).lower() in NEXT_TEXTS:
                href = a["href"]
                if href and href != current_url and "javascript" not in href.lower():
                    return href if href.startswith("http") else urljoin(current_url, href)

        # Numeric vBulletin pagination
        page_match = re.search(r"[/=](\d+)(?:\.html)?$", current_url)
        cur_page = int(page_match.group(1)) if page_match else 1
        nxt = cur_page + 1
        for a in soup.select("a[href]"):
            href = a["href"]
            if f"page{nxt}" in href or f"={nxt}" in href or f"/{nxt}" in href:
                return href if href.startswith("http") else urljoin(current_url, href)
    except Exception as e:
        print(f"[teambhp] next-page detection error: {e}")
    return None


# ── nodriver async core ────────────────────────────────────────────────────────

async def _scrape_with_nodriver(url: str, max_pages: int) -> dict:
    """
    Main scraping coroutine using nodriver.
    Chrome opens as a real (visible) window — this is intentional.
    Cloudflare Bot Management cannot detect it without the headless flag and
    standard automation fingerprints.
    """
    import nodriver as nd  # type: ignore[import-untyped]

    thread_id = extract_thread_id(url)
    all_posts: List[Dict] = []
    thread_title = ""

    # Start a real Chrome process (no --headless, no patches)
    browser = await nd.start(
        headless=False,           # visible window bypasses CF fingerprinting
        browser_args=[
            "--window-size=1280,900",
            "--lang=en-US",
        ],
    )

    try:
        current_url = url
        page_num = 1

        while page_num <= max_pages:
            print(f"[teambhp] nodriver — page {page_num}: {current_url}")
            tab = await browser.get(current_url)

            # Wait up to 15 s for Cloudflare challenge to clear
            for attempt in range(15):
                await asyncio.sleep(1)
                try:
                    title = await tab.evaluate("document.title")
                    html_check = await tab.evaluate("document.documentElement.outerHTML")
                except Exception:
                    await asyncio.sleep(1)
                    continue
                if not _is_cloudflare_block(html_check[:2000], title):
                    break
                print(f"[teambhp] Cloudflare challenge active, waiting... ({attempt+1}s)")

            # Extra settle time after CF clears
            await asyncio.sleep(random.uniform(1.5, 3.0))

            html = await tab.evaluate("document.documentElement.outerHTML")
            title = await tab.evaluate("document.title")

            if _is_cloudflare_block(html[:3000], title):
                return {
                    "success": False,
                    "error": (
                        "Cloudflare is blocking access even after waiting. "
                        "Try opening team-bhp.com in Chrome manually first to set a "
                        "valid CF cookie, then retry."
                    ),
                    "posts": [],
                }

            if page_num == 1:
                thread_title = _get_thread_title(html, title)

            page_posts = _parse_posts_from_html(html)
            print(f"[teambhp] Page {page_num}: {len(page_posts)} posts parsed")
            if page_posts:
                sample = page_posts[0]
                print(f"[teambhp] Sample post — author: '{sample['author']}' | "
                      f"text[:120]: {repr(sample['text'][:120])}")

            if not page_posts:
                if page_num == 1:
                    print(f"[teambhp] DEBUG — HTML snippet:\n{html[:3000]}")
                    return {
                        "success": False,
                        "error": (
                            "Page loaded but no posts found. "
                            "HTML structure may have changed — check DEBUG output above."
                        ),
                        "posts": [],
                    }
                # Subsequent page returned nothing — end of thread or transient block.
                print(f"[teambhp] Page {page_num} returned 0 posts — stopping pagination.")
                break

            all_posts.extend(page_posts)

            next_url = _get_next_page_url(html, current_url)
            if not next_url or next_url == current_url:
                print(f"[teambhp] No next-page link found — end of thread at page {page_num}.")
                break

            current_url = next_url
            page_num += 1

            # Human-like delay between page loads
            await asyncio.sleep(random.uniform(2.0, 4.0))

    finally:
        try:
            browser.stop()
        except Exception:
            pass

    print(f"[teambhp] Done — {len(all_posts)} posts, thread: '{thread_title}'")
    return {
        "success": True,
        "thread_id": thread_id,
        "thread_title": thread_title or f"TeamBHP Thread {thread_id}",
        "url": url,
        "posts": all_posts,
    }


# ── Public sync API ────────────────────────────────────────────────────────────

def scrape_teambhp_thread(url: str, max_pages: int = 15) -> dict:
    """
    Scrape a TeamBHP forum thread.  Returns:

        {
            "success"      : bool,
            "thread_id"    : str,
            "thread_title" : str,
            "url"          : str,
            "posts"        : [{"post_id", "author", "date", "text"}, ...],
            "error"        : str   # only on failure
        }

    Runs the async nodriver scraper in a fresh event loop so it is safe to call
    from synchronous FastAPI route handlers.
    """
    try:
        return asyncio.run(_scrape_with_nodriver(url, max_pages))
    except ModuleNotFoundError as exc:
        if "nodriver" in str(exc):
            return {
                "success": False,
                "error": "nodriver is not installed. Run: pip install nodriver",
                "posts": [],
            }
        import traceback
        print(f"[teambhp] Scrape error:\n{traceback.format_exc()}")
        return {"success": False, "error": str(exc), "posts": []}
    except Exception as exc:
        import traceback
        print(f"[teambhp] Scrape error:\n{traceback.format_exc()}")
        return {"success": False, "error": str(exc), "posts": []}
