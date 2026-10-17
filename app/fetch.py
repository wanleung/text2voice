"""Fetch a web article and extract its readable text."""
import asyncio
import difflib
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura

MAX_BYTES = 5_000_000
MAX_REDIRECTS = 5
USER_AGENT = "Mozilla/5.0 (compatible; text2voice/0.1; +article-to-speech)"


class FetchError(ValueError):
    pass


@dataclass
class Article:
    title: str
    text: str  # Markdown-ish: "# title" then paragraphs separated by blank lines
    url: str


async def _check_public(url: str) -> None:
    """Refuse URLs that resolve to private, loopback or link-local addresses (SSRF guard)."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("only http(s) URLs are supported")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, parts.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise FetchError(f"cannot resolve {parts.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise FetchError(f"{parts.hostname} resolves to a non-public address")


async def _download(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await _check_public(url)  # re-checked on every redirect hop
            async with client.stream("GET", url) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code >= 400:
                    raise FetchError(f"{url} returned HTTP {r.status_code}")
                body = bytearray()
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        raise FetchError("page is larger than 5 MB")
                return bytes(body).decode(r.encoding or "utf-8", errors="replace"), str(r.url)
    raise FetchError("too many redirects")


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).casefold()


def _dedupe(blocks: list[str], seen: list[str]) -> list[str]:
    """Drop repeated paragraphs. Some sites render the article twice (desktop + mobile), and the
    copies can differ by a word, so near-identical paragraphs count as repeats too."""
    kept: list[str] = []
    norms = [_norm(x) for x in seen if _norm(x)]
    for block in blocks:
        n = _norm(block)
        if not n:
            continue
        if any(n == m or (len(n) > 15 and difflib.SequenceMatcher(None, n, m).ratio() > 0.9) for m in norms):
            continue
        kept.append(block)
        norms.append(n)
    return kept


def extract(html: str, url: str) -> Article:
    body = trafilatura.extract(html, url=url, output_format="markdown", include_formatting=True,
                               include_comments=False, include_tables=False)
    if not body:
        raise FetchError("could not find article text on the page")
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta and meta.title else "").strip()
    # Code isn't worth reading aloud: fenced blocks, and one-line <pre> rendered as a lone `span`.
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    body = re.sub(r"^[ \t]*`[^`\n]+`[ \t]*$", "", body, flags=re.M)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body)]
    blocks = [b for b in blocks if b]
    blocks = _dedupe(blocks, seen=[title])  # also drops body headings that repeat the title
    if not blocks:
        raise FetchError("could not find article text on the page")
    text = "\n\n".join(blocks)
    return Article(title=title, text=f"# {title}\n\n{text}" if title else text, url=url)


async def fetch_article(url: str) -> Article:
    html, final_url = await _download(url)
    return await asyncio.to_thread(extract, html, final_url)
