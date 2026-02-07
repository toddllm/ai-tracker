from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import os
import queue
import re
import shlex
import select
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from time import struct_time
from typing import Any
import termios
import tty
import webbrowser

import feedparser
import requests
from dateutil import parser as date_parser
from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ARXIV_API_URL = "https://export.arxiv.org/api/query"
X_SEARCH_ENDPOINTS = [
    "https://api.x.com/2/tweets/search/recent",
    "https://api.twitter.com/2/tweets/search/recent",
]
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
DEFAULT_TOPICS = "agents, open-source models, multimodal, reasoning"

NEWSLETTER_FEEDS: dict[str, str] = {
    "Import AI (Substack)": "https://importai.substack.com/feed",
    "Latent Space": "https://www.latent.space/feed",
    "Last Week in AI": "https://lastweekin.ai/feed",
    "SemiAnalysis": "https://newsletter.semianalysis.com/feed",
    "Interconnects": "https://www.interconnects.ai/feed",
    "Understanding AI": "https://www.understandingai.org/feed",
    "AI Tidbits": "https://www.aitidbits.ai/feed",
    "One Useful Thing": "https://www.oneusefulthing.org/feed",
}

VALID_SOURCES = {"x": "X", "arxiv": "arXiv", "newsletters": "Newsletters"}
SECTION_ORDER = (
    "status",
    "brief",
    "feed",
    "story",
    "models",
    "sources",
    "menu",
    "commands",
)
RIGHT_SECTIONS = {"story", "models", "sources", "menu", "commands"}

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AppConfig:
    topics: str
    sources: list[str]
    strict_topics: bool
    lookback_days: int
    max_items: int
    sort_mode: str
    newsletters: tuple[str, ...]
    quick_filter: str
    refresh_minutes: int
    ollama_host: str
    ollama_model: str
    analysis_limit: int
    ollama_timeout_seconds: int
    model_list_size: int
    model_poll_seconds: int
    once: bool


@dataclass
class RuntimeState:
    cycle_state: dict[str, Any]
    next_run_at: datetime
    is_refreshing: bool
    status_message: str
    command_log: list[str]
    selected_index: int
    right_column_ratio: int
    brief_focus: bool
    brief_scroll: int
    active_section: str
    status_row_ratio: int
    brief_row_ratio: int
    body_row_ratio: int
    show_menu: bool
    in_command_mode: bool
    command_buffer: str


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        raw = " ".join(str(piece) for piece in raw)
    if not isinstance(raw, str):
        raw = str(raw)
    unescaped = html.unescape(raw)
    no_html = HTML_TAG_RE.sub(" ", unescaped)
    return WHITESPACE_RE.sub(" ", no_html).strip()


def parse_date(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        parsed = raw
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if isinstance(raw, (tuple, struct_time)):
        try:
            parsed = datetime(*list(raw)[:6], tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None
    try:
        parsed = date_parser.parse(str(raw))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_keywords(raw_keywords: str) -> list[str]:
    return [piece.strip() for piece in raw_keywords.split(",") if piece.strip()]


def matches_keywords(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def recency_score(published_at: datetime) -> float:
    age_hours = max((now_utc() - published_at).total_seconds() / 3600.0, 1.0)
    return 1.0 / (1.0 + math.log(age_hours + 1.0))


def age_hours_from_now(published_at: datetime) -> float:
    return max((now_utc() - published_at).total_seconds() / 3600.0, 0.0)


def recency_priority_score(published_at: datetime) -> float:
    # Aggressive recency preference so very recent items surface in trending.
    age_hours = age_hours_from_now(published_at)
    base = math.exp(-age_hours / 36.0)
    if age_hours <= 6:
        base += 0.22
    elif age_hours <= 24:
        base += 0.10
    return max(0.02, min(1.0, base))


def dedupe_key(item: dict[str, Any]) -> str:
    return item.get("url", "") or f'{item["source"]}:{item["title"][:80]}'


def enforce_recent_source_mix(
    items: list[dict[str, Any]],
    required_sources: tuple[str, ...] = ("X", "Newsletter", "arXiv"),
    recent_hours: int = 72,
    top_slots: int = 6,
) -> list[dict[str, Any]]:
    if not items:
        return items

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for source in required_sources:
        candidates = [
            item
            for item in items
            if item.get("source") == source and age_hours_from_now(item["published_at"]) <= recent_hours
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                item["published_at"],
                float(item.get("trend_score", item.get("score", 0.0))),
            ),
            reverse=True,
        )
        candidate = candidates[0]
        key = dedupe_key(candidate)
        if key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(key)

    if not selected:
        return items

    selected = selected[: max(1, top_slots)]
    selected.sort(
        key=lambda item: (
            item["published_at"],
            float(item.get("trend_score", item.get("score", 0.0))),
        ),
        reverse=True,
    )
    remainder = [item for item in items if dedupe_key(item) not in selected_keys]
    return selected + remainder


def make_item(
    source: str,
    title: str,
    url: str,
    summary: str,
    published_at: datetime,
    author: str,
    score: float,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "title": normalize_text(title),
        "url": url,
        "summary": normalize_text(summary),
        "published_at": published_at,
        "author": normalize_text(author) or "Unknown",
        "score": float(score),
        "metrics": metrics or {},
        "llm": None,
    }


def dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        url = item.get("url", "")
        dedupe_key = url or f'{item["source"]}:{item["title"][:80]}'
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
    return deduped


def fetch_arxiv_items(
    keywords_csv: str,
    max_items: int,
    days_back: int,
    strict_topics: bool,
) -> list[dict[str, Any]]:
    keywords = parse_keywords(keywords_csv)
    use_topic_query = strict_topics and bool(keywords)
    if use_topic_query:
        query = " OR ".join([f'all:"{keyword}"' for keyword in keywords])
    else:
        query = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": min(max(max_items * 4, 60), 240),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=25)
        response.raise_for_status()
    except requests.RequestException:
        return []

    parsed = feedparser.parse(response.text)
    cutoff = now_utc() - timedelta(days=days_back)
    items: list[dict[str, Any]] = []
    local_limit = max_items * 3

    for entry in parsed.entries:
        published = parse_date(entry.get("published") or entry.get("updated"))
        if not published or published < cutoff:
            continue

        title = normalize_text(entry.get("title", ""))
        summary = normalize_text(entry.get("summary", ""))
        if strict_topics and keywords and not matches_keywords(f"{title} {summary}", keywords):
            continue

        authors = ", ".join(author.get("name", "") for author in entry.get("authors", []))
        items.append(
            make_item(
                source="arXiv",
                title=title,
                url=entry.get("link", ""),
                summary=summary,
                published_at=published,
                author=authors or "arXiv",
                score=recency_score(published),
            )
        )
        if len(items) >= local_limit:
            break

    items.sort(key=lambda item: item["published_at"], reverse=True)
    return items[:local_limit]


def fetch_newsletter_items(
    selected_newsletters: tuple[str, ...],
    keywords_csv: str,
    max_items: int,
    days_back: int,
    strict_topics: bool,
) -> list[dict[str, Any]]:
    if not selected_newsletters:
        return []

    keywords = parse_keywords(keywords_csv)
    cutoff = now_utc() - timedelta(days=days_back)
    per_feed_limit = max(
        12,
        min(80, math.ceil(max_items / max(1, len(selected_newsletters))) * 5),
    )
    local_limit = max_items * 4

    items: list[dict[str, Any]] = []
    for newsletter_name in selected_newsletters:
        feed_url = NEWSLETTER_FEEDS.get(newsletter_name)
        if not feed_url:
            continue

        parsed = feedparser.parse(feed_url)
        if parsed.bozo and not parsed.entries:
            continue

        for entry in parsed.entries[: per_feed_limit * 2]:
            published = parse_date(
                entry.get("published")
                or entry.get("updated")
                or entry.get("published_parsed")
                or entry.get("updated_parsed")
            )
            if not published or published < cutoff:
                continue

            title = normalize_text(entry.get("title", ""))
            summary = normalize_text(entry.get("summary", "") or entry.get("description", ""))
            if strict_topics and keywords and not matches_keywords(f"{title} {summary}", keywords):
                continue

            author = normalize_text(
                entry.get("author") or parsed.feed.get("title") or newsletter_name
            )
            items.append(
                make_item(
                    source="Newsletter",
                    title=title,
                    url=entry.get("link", ""),
                    summary=summary,
                    published_at=published,
                    author=author,
                    score=recency_score(published),
                )
            )
            if len(items) >= local_limit:
                break

    items.sort(key=lambda item: item["published_at"], reverse=True)
    return items[:local_limit]


def fetch_x_items(
    keywords_csv: str,
    max_items: int,
    days_back: int,
    strict_topics: bool,
) -> dict[str, Any]:
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        return {
            "items": [],
            "error": "X source disabled. Set X_BEARER_TOKEN to enable X API access.",
        }

    keywords = parse_keywords(keywords_csv)
    if strict_topics and keywords:
        keyword_query = " OR ".join(
            f'"{keyword}"' if " " in keyword else keyword for keyword in keywords
        )
    else:
        keyword_query = "AI OR LLM OR \"machine learning\" OR arXiv OR agents"

    query = f"({keyword_query}) lang:en -is:retweet -is:reply"
    params = {
        "query": query,
        "max_results": min(max(max_items * 2, 20), 100),
        "tweet.fields": "created_at,public_metrics,lang",
        "user.fields": "name,username",
        "expansions": "author_id",
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}

    response_payload: dict[str, Any] | None = None
    last_error = "Unknown error"
    for endpoint in X_SEARCH_ENDPOINTS:
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=20)
        except requests.RequestException as exc:
            last_error = str(exc)
            continue

        if response.status_code == 200:
            response_payload = response.json()
            break

        preview = normalize_text(response.text)[:140]
        last_error = f"HTTP {response.status_code}: {preview}"

    if response_payload is None:
        return {
            "items": [],
            "error": f"X source unavailable ({last_error}).",
        }

    users = {
        user.get("id", ""): user
        for user in response_payload.get("includes", {}).get("users", [])
    }
    cutoff = now_utc() - timedelta(days=min(7, days_back))
    items: list[dict[str, Any]] = []
    local_limit = max_items * 3

    for tweet in response_payload.get("data", []):
        created_at = parse_date(tweet.get("created_at"))
        if not created_at or created_at < cutoff:
            continue

        text = normalize_text(tweet.get("text", ""))
        if strict_topics and keywords and not matches_keywords(text, keywords):
            continue

        author_data = users.get(tweet.get("author_id", ""), {})
        username = author_data.get("username", "x")
        display_name = author_data.get("name") or username
        metrics = tweet.get("public_metrics", {}) or {}
        engagement = (
            metrics.get("retweet_count", 0) * 2.0
            + metrics.get("quote_count", 0) * 2.5
            + metrics.get("reply_count", 0) * 1.5
            + metrics.get("like_count", 0)
        )
        score = (engagement + 1.0) * recency_score(created_at)
        title = text if len(text) <= 150 else f"{text[:147]}..."

        items.append(
            make_item(
                source="X",
                title=title,
                url=f"https://x.com/{username}/status/{tweet.get('id', '')}",
                summary=text,
                published_at=created_at,
                author=f"{display_name} (@{username})",
                score=score,
                metrics=metrics,
            )
        )
        if len(items) >= local_limit:
            break

    return {"items": items, "error": ""}


def build_combined_feed(
    source_selection: list[str],
    keywords: str,
    max_items: int,
    days_back: int,
    newsletter_selection: tuple[str, ...],
    strict_topics: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    feed_items: list[dict[str, Any]] = []
    errors: list[str] = []

    if "arXiv" in source_selection:
        try:
            feed_items.extend(
                fetch_arxiv_items(
                    keywords_csv=keywords,
                    max_items=max_items,
                    days_back=days_back,
                    strict_topics=strict_topics,
                )
            )
        except Exception as exc:  # pragma: no cover
            errors.append(f"arXiv source unavailable ({normalize_text(str(exc))[:120]}).")

    if "Newsletters" in source_selection:
        try:
            feed_items.extend(
                fetch_newsletter_items(
                    selected_newsletters=newsletter_selection,
                    keywords_csv=keywords,
                    max_items=max_items,
                    days_back=days_back,
                    strict_topics=strict_topics,
                )
            )
        except Exception as exc:  # pragma: no cover
            errors.append(
                f"Newsletter feeds unavailable ({normalize_text(str(exc))[:120]})."
            )

    if "X" in source_selection:
        x_result = fetch_x_items(
            keywords_csv=keywords,
            max_items=max_items,
            days_back=days_back,
            strict_topics=strict_topics,
        )
        feed_items.extend(x_result["items"])
        if x_result["error"]:
            errors.append(x_result["error"])

    return dedupe_items(feed_items), errors


def sanitize_ollama_host(host: str) -> str:
    return host.strip().rstrip("/")


def humanize_bytes(num: int | float | None) -> str:
    if not num:
        return "-"
    value = float(num)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return "-"


def human_age(published_at: datetime) -> str:
    delta = now_utc() - published_at
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


def fetch_ollama_models(ollama_host: str) -> dict[str, Any]:
    base = sanitize_ollama_host(ollama_host)
    if not base:
        return {"models": [], "error": "Missing Ollama host URL."}
    try:
        response = requests.get(f"{base}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return {
            "models": [],
            "error": f"Ollama unavailable at {base}. Start it with: ollama serve",
        }

    models: list[dict[str, Any]] = []
    for model in payload.get("models", []):
        model_name = normalize_text(model.get("name", ""))
        if not model_name:
            continue
        size_bytes = int(model.get("size", 0) or 0)
        modified_at = normalize_text(model.get("modified_at", ""))
        models.append(
            {
                "name": model_name,
                "size_bytes": size_bytes,
                "modified_at": modified_at,
            }
        )

    models.sort(key=lambda m: (m["size_bytes"], m["name"]), reverse=True)
    return {"models": models, "error": ""}


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def ollama_rank_items(
    ollama_host: str,
    ollama_model: str,
    topics_csv: str,
    serialized_items: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    base = sanitize_ollama_host(ollama_host)
    if not base:
        return {"analysis": {}, "brief": "", "error": "Missing Ollama host."}
    if not ollama_model:
        return {"analysis": {}, "brief": "", "error": "No Ollama model selected."}

    topics = parse_keywords(topics_csv)
    topic_text = ", ".join(topics) if topics else "general AI updates"

    prompt = (
        "You are an AI news analyst. Rank and summarize feed items for a technical reader.\\n"
        f"Priority topics: {topic_text}\\n"
        "Return only a single JSON object, with no markdown or commentary.\\n"
        "Required JSON schema:\\n"
        "{\\n"
        "  \\\"items\\\": [\\n"
        "    {\\n"
        "      \\\"id\\\": \\\"item-1\\\",\\n"
        "      \\\"relevance\\\": 0-10 number,\\n"
        "      \\\"novelty\\\": 0-10 number,\\n"
        "      \\\"impact\\\": 0-10 number,\\n"
        "      \\\"editor_score\\\": 0-10 number (what a strong editor should surface),\\n"
        "      \\\"freshness_priority\\\": 0-10 number (prefer breaking/recent items),\\n"
        "      \\\"summary\\\": \\\"<=35 words\\\",\\n"
        "      \\\"tags\\\": [\\\"tag1\\\", \\\"tag2\\\", \\\"tag3\\\"],\\n"
        "      \\\"reason\\\": \\\"<=18 words\\\"\\n"
        "    }\\n"
        "  ]\\n"
        "}\\n"
        "Do not include a long narrative in this JSON response.\\n"
        "Use all provided ids exactly once when possible.\\n"
        f"Feed items:\\n{serialized_items}"
    )

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 2000,
        },
    }
    def call_generate(request_payload: dict[str, Any]) -> tuple[str, str]:
        try:
            response = requests.post(
                f"{base}/api/generate",
                json=request_payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            return "", f"Ollama call failed ({normalize_text(str(exc))[:160]})."
        return str(result.get("response", "")), ""

    raw_response, error = call_generate(payload)
    if error:
        return {
            "analysis": {},
            "brief": "",
            "error": error,
        }

    parsed = extract_json_object(raw_response)
    if not parsed:
        parsed = extract_json_object(normalize_text(raw_response))

    if not parsed:
        retry_prompt = (
            "Return JSON only. If uncertain, still return valid JSON with empty items. "
            "Schema: {\"items\":[{\"id\":\"item-1\",\"relevance\":0,\"novelty\":0,"
            "\"impact\":0,\"editor_score\":0,\"freshness_priority\":0,"
            "\"summary\":\"\",\"tags\":[],\"reason\":\"\"}]}.\n"
            f"Feed items:\n{serialized_items}"
        )
        retry_payload = {
            "model": ollama_model,
            "prompt": retry_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_predict": 1000,
            },
        }
        retry_raw, retry_error = call_generate(retry_payload)
        if not retry_error:
            parsed = extract_json_object(retry_raw)
            if not parsed:
                parsed = extract_json_object(normalize_text(retry_raw))
            if not parsed and retry_raw.strip():
                return {
                    "analysis": {},
                    "brief": retry_raw.strip(),
                    "error": "",
                }

    if not parsed:
        fallback_brief = raw_response.strip()
        return {
            "analysis": {},
            "brief": fallback_brief,
            "error": "",
        }
    analysis: dict[str, dict[str, Any]] = {}
    for entry in parsed.get("items", []):
        item_id = normalize_text(entry.get("id"))
        if not item_id:
            continue
        analysis[item_id] = {
            "relevance": max(0.0, min(10.0, float(entry.get("relevance", 0) or 0))),
            "novelty": max(0.0, min(10.0, float(entry.get("novelty", 0) or 0))),
            "impact": max(0.0, min(10.0, float(entry.get("impact", 0) or 0))),
            "editor_score": max(0.0, min(10.0, float(entry.get("editor_score", 0) or 0))),
            "freshness_priority": max(
                0.0, min(10.0, float(entry.get("freshness_priority", 0) or 0))
            ),
            "summary": normalize_text(entry.get("summary", "")),
            "tags": [normalize_text(tag) for tag in entry.get("tags", []) if normalize_text(tag)],
            "reason": normalize_text(entry.get("reason", "")),
        }

    brief_value = ""
    for key in ("brief_markdown", "brief", "summary", "content"):
        candidate = parsed.get(key)
        if isinstance(candidate, str) and candidate.strip():
            brief_value = candidate.strip()
            break

    return {
        "analysis": analysis,
        "brief": brief_value,
        "error": "",
    }


def ollama_editor_brief_markdown(
    ollama_host: str,
    ollama_model: str,
    topics_csv: str,
    serialized_items: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    base = sanitize_ollama_host(ollama_host)
    if not base:
        return {"brief_markdown": "", "error": "Missing Ollama host."}
    if not ollama_model:
        return {"brief_markdown": "", "error": "No Ollama model selected."}

    topics = parse_keywords(topics_csv)
    topic_text = ", ".join(topics) if topics else "general AI updates"
    prompt = (
        "You are an AI news editor. Write a detailed markdown briefing from the feed items.\n"
        f"Priority topics: {topic_text}\n"
        "Output markdown only (no JSON).\n"
        "Use this structure exactly:\n"
        "# AI Briefing\n"
        "## Top Developments\n"
        "- 4-8 bullet points\n"
        "## Source Highlights\n"
        "### X\n"
        "### arXiv\n"
        "### Newsletters\n"
        "## Editor Notes\n"
        "- Why this matters now\n"
        "- What to watch next\n"
        "Length target: 500-1400 words.\n"
        "Use concise paragraphs, bullet lists, and explicit links when available.\n"
        "Finish with complete sentences; do not cut off the final section.\n"
        f"Feed items:\n{serialized_items}"
    )
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 6000,
        },
    }
    try:
        response = requests.post(
            f"{base}/api/generate",
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        return {
            "brief_markdown": "",
            "error": f"Ollama brief failed ({normalize_text(str(exc))[:160]}).",
        }

    raw_response = str(result.get("response", "")).strip()
    if not raw_response:
        return {"brief_markdown": "", "error": "Ollama brief was empty."}

    parsed = extract_json_object(raw_response)
    if parsed:
        for key in ("brief_markdown", "brief", "summary", "content"):
            candidate = parsed.get(key)
            normalized = normalize_markdown_text(candidate)
            if normalized:
                return {"brief_markdown": normalized, "error": ""}
    return {"brief_markdown": normalize_markdown_text(raw_response), "error": ""}


def apply_ollama_enrichment(
    items: list[dict[str, Any]],
    ollama_host: str,
    ollama_model: str,
    topics_csv: str,
    analysis_limit: int,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], str, str]:
    if not items:
        return items, "", ""
    if analysis_limit <= 0:
        return items, "", ""

    capped = min(len(items), analysis_limit)
    candidate_items = sorted(items, key=lambda item: item["published_at"], reverse=True)[:capped]
    payload = []
    id_to_dedupe_key: dict[str, str] = {}
    for index, item in enumerate(candidate_items, start=1):
        item_id = f"item-{index}"
        dedupe_key = item.get("url", "") or f'{item["source"]}:{item["title"][:80]}'
        id_to_dedupe_key[item_id] = dedupe_key
        payload.append(
            {
                "id": item_id,
                "source": item["source"],
                "title": item["title"][:220],
                "summary": item["summary"][:700],
                "author": item["author"],
                "published_at": item["published_at"].isoformat(),
            }
        )
    serialized_payload = json.dumps(payload, ensure_ascii=True)

    ranking_result = ollama_rank_items(
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        topics_csv=topics_csv,
        serialized_items=serialized_payload,
        timeout_seconds=timeout_seconds,
    )
    if ranking_result["error"]:
        return items, "", ranking_result["error"]

    analysis_map = ranking_result["analysis"]
    dedupe_key_to_llm: dict[str, dict[str, Any]] = {}
    for item_id, llm_data in analysis_map.items():
        dedupe_key = id_to_dedupe_key.get(item_id)
        if dedupe_key:
            dedupe_key_to_llm[dedupe_key] = llm_data

    updated: list[dict[str, Any]] = []
    for item in items:
        cloned = dict(item)
        dedupe_key = item.get("url", "") or f'{item["source"]}:{item["title"][:80]}'
        llm_data = dedupe_key_to_llm.get(dedupe_key)
        if llm_data:
            llm_bonus = (
                llm_data["relevance"] * 0.30
                + llm_data["novelty"] * 0.20
                + llm_data["impact"] * 0.20
                + llm_data["editor_score"] * 0.20
                + llm_data["freshness_priority"] * 0.10
            ) / 10.0
            cloned["score"] = float(cloned["score"]) + llm_bonus
            cloned["llm"] = llm_data
        updated.append(cloned)

    editor_brief = ollama_editor_brief_markdown(
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        topics_csv=topics_csv,
        serialized_items=serialized_payload,
        timeout_seconds=timeout_seconds,
    )
    brief_markdown = editor_brief["brief_markdown"] or ranking_result["brief"]
    if editor_brief["error"]:
        return updated, brief_markdown, editor_brief["error"]

    return updated, brief_markdown, ""


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return f"{value[: width - 1]}…"


def parse_sources(raw: str) -> list[str]:
    if not raw.strip():
        return ["Newsletters", "arXiv", "X"]
    out: list[str] = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in VALID_SOURCES:
            valid = ", ".join(VALID_SOURCES.keys())
            raise ValueError(f"Unknown source '{part.strip()}'. Valid sources: {valid}")
        canonical = VALID_SOURCES[key]
        if canonical not in out:
            out.append(canonical)
    return out


def parse_newsletters(raw: str) -> tuple[str, ...]:
    if not raw.strip() or raw.strip().lower() == "all":
        return tuple(NEWSLETTER_FEEDS.keys())
    chosen: list[str] = []
    names_by_lower = {name.lower(): name for name in NEWSLETTER_FEEDS}
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in names_by_lower:
            valid = ", ".join(NEWSLETTER_FEEDS.keys())
            raise ValueError(f"Unknown newsletter '{part.strip()}'. Valid: {valid}")
        resolved = names_by_lower[key]
        if resolved not in chosen:
            chosen.append(resolved)
    return tuple(chosen)


def apply_quick_filter(items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    if not query.strip():
        return items
    lowered = query.lower()
    return [
        item
        for item in items
        if lowered in item["title"].lower() or lowered in item["summary"].lower()
    ]


def run_cycle(config: AppConfig) -> dict[str, Any]:
    started = now_utc()
    errors: list[str] = []

    models_result = fetch_ollama_models(config.ollama_host)
    models = models_result["models"]
    model_error = models_result["error"]

    items, source_errors = build_combined_feed(
        source_selection=config.sources,
        keywords=config.topics,
        max_items=config.max_items,
        days_back=config.lookback_days,
        newsletter_selection=config.newsletters,
        strict_topics=config.strict_topics,
    )
    errors.extend(source_errors)

    items = apply_quick_filter(items, config.quick_filter)
    items.sort(key=lambda item: item["published_at"], reverse=True)

    llm_brief = ""
    model_names = {m["name"] for m in models}
    if config.analysis_limit > 0:
        if config.ollama_model not in model_names:
            errors.append(
                f"Ollama model '{config.ollama_model}' not found locally. Run: ollama pull {config.ollama_model}"
            )
        else:
            items, llm_brief, llm_error = apply_ollama_enrichment(
                items=items,
                ollama_host=config.ollama_host,
                ollama_model=config.ollama_model,
                topics_csv=config.topics,
                analysis_limit=config.analysis_limit,
                timeout_seconds=config.ollama_timeout_seconds,
            )
            if llm_error:
                errors.append(llm_error)

    if config.sort_mode == "trending":
        for item in items:
            llm = item.get("llm") or {}
            base = float(item.get("score", 0.0))
            recency = recency_priority_score(item["published_at"])
            editor_score = float(llm.get("editor_score", 0.0)) / 10.0
            freshness_pref = float(llm.get("freshness_priority", 0.0)) / 10.0
            item["trend_score"] = (
                base * 0.40
                + recency * 0.45
                + editor_score * 0.10
                + freshness_pref * 0.05
            )
        items.sort(
            key=lambda item: (
                float(item.get("trend_score", item.get("score", 0.0))),
                item["published_at"],
            ),
            reverse=True,
        )
        items = enforce_recent_source_mix(items, top_slots=min(6, max(3, config.max_items // 4)))
    else:
        items.sort(key=lambda item: item["published_at"], reverse=True)

    elapsed_seconds = max((now_utc() - started).total_seconds(), 0.0)
    return {
        "started": started,
        "elapsed_seconds": elapsed_seconds,
        "items": items[: config.max_items],
        "errors": errors,
        "llm_brief": llm_brief,
        "models": models,
        "model_error": model_error,
    }


def make_initial_cycle_state(config: AppConfig) -> dict[str, Any]:
    return {
        "started": now_utc(),
        "elapsed_seconds": 0.0,
        "items": [],
        "errors": [],
        "llm_brief": "",
        "models": [],
        "model_error": "Discovering local Ollama models...",
    }


def clamp_selection(index: int, items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    if index < 0:
        return 0
    if index >= len(items):
        return len(items) - 1
    return index


def cycle_selection(index: int, items: list[dict[str, Any]], delta: int) -> int:
    if not items:
        return 0
    return (index + delta) % len(items)


def clamp_right_column_ratio(value: int) -> int:
    return max(2, min(6, value))


def clamp_row_ratio(value: int) -> int:
    return max(1, min(12, value))


def cycle_section(current: str, step: int) -> str:
    if current not in SECTION_ORDER:
        return SECTION_ORDER[0]
    index = SECTION_ORDER.index(current)
    return SECTION_ORDER[(index + step) % len(SECTION_ORDER)]


def section_to_row(section: str) -> str:
    if section == "status":
        return "status"
    if section == "brief":
        return "brief"
    return "body"


def adjust_focused_row_height(runtime_state: RuntimeState, delta: int) -> str:
    row_key = section_to_row(runtime_state.active_section)
    if row_key == "status":
        runtime_state.status_row_ratio = clamp_row_ratio(runtime_state.status_row_ratio + delta)
        return f"status_row_ratio -> {runtime_state.status_row_ratio}"
    if row_key == "brief":
        runtime_state.brief_row_ratio = clamp_row_ratio(runtime_state.brief_row_ratio + delta)
        return f"brief_row_ratio -> {runtime_state.brief_row_ratio}"
    runtime_state.body_row_ratio = clamp_row_ratio(runtime_state.body_row_ratio + delta)
    return f"body_row_ratio -> {runtime_state.body_row_ratio}"


def open_link(url: str) -> str:
    clean_url = url.strip()
    if not clean_url:
        return "No URL available for selected story."
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", clean_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            webbrowser.open(clean_url, new=2)
        return ""
    except Exception as exc:
        return f"Failed to open link: {exc}"


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def normalize_markdown_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        parts = [normalize_markdown_text(part) for part in raw]
        return "\n".join(part for part in parts if part).strip()
    text = raw if isinstance(raw, str) else str(raw)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return strip_code_fence(text).strip()


def format_llm_brief_markdown(brief: str) -> str:
    cleaned = normalize_markdown_text(brief)
    if not cleaned:
        return "_No brief available yet._"

    parsed = extract_json_object(cleaned)
    if parsed:
        for key in ("brief_markdown", "brief", "summary", "content"):
            extracted = normalize_markdown_text(parsed.get(key))
            if extracted:
                return extracted

    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError:
        raw = None
    if isinstance(raw, str):
        extracted = normalize_markdown_text(raw)
        if extracted:
            return extracted
    elif isinstance(raw, dict):
        for key in ("brief_markdown", "brief", "summary", "content"):
            extracted = normalize_markdown_text(raw.get(key))
            if extracted:
                return extracted

    return cleaned


@lru_cache(maxsize=32)
def render_markdown_lines(markdown_text: str, width: int) -> tuple[str, ...]:
    render_width = max(40, width)
    render_console = Console(
        width=render_width,
        force_terminal=False,
        color_system=None,
    )
    with render_console.capture() as capture:
        render_console.print(Markdown(markdown_text))
    rendered = capture.get().strip("\n")
    if not rendered:
        return ("",)
    return tuple(rendered.splitlines())


def build_brief_panel(
    llm_brief: str,
    brief_focus: bool,
    brief_scroll: int,
    terminal_width: int,
    terminal_height: int,
    focused: bool = False,
) -> tuple[Panel, int, int]:
    border_style = "bright_green" if focused else "green"
    markdown_text = format_llm_brief_markdown(llm_brief)
    if markdown_text == "_No brief available yet._":
        return (
            Panel(Markdown(markdown_text), title="Ollama Brief", border_style=border_style),
            0,
            0,
        )

    content_width = max(
        50,
        terminal_width - 12 if brief_focus else (terminal_width // 2) - 10,
    )
    rendered_lines = list(render_markdown_lines(markdown_text, content_width))
    if not rendered_lines:
        rendered_lines = [""]

    visible_lines = (
        max(10, terminal_height - 18)
        if brief_focus
        else max(6, min(18, terminal_height // 4))
    )
    max_scroll = max(0, len(rendered_lines) - visible_lines)
    offset = max(0, min(brief_scroll, max_scroll))
    window = rendered_lines[offset : offset + visible_lines]
    line_end = offset + len(window)
    footer = f"Lines {offset + 1}-{line_end}/{len(rendered_lines)}"
    if max_scroll > 0:
        footer += " | PgUp/PgDn scroll"
        if brief_focus:
            footer += " | Up/Down scroll in focus mode"

    body = "\n".join(window).rstrip()
    if body:
        body += "\n\n"
    body += footer
    return Panel(body, title="Ollama Brief", border_style=border_style), max_scroll, offset


def format_story_markdown(item: dict[str, Any]) -> str:
    url = item.get("url", "").strip()
    title = item.get("title", "(untitled)")
    published = item.get("published_at")
    published_text = (
        published.strftime("%Y-%m-%d %H:%M UTC")
        if isinstance(published, datetime)
        else "unknown"
    )
    source = item.get("source", "?")
    author = item.get("author", "Unknown")
    score = float(item.get("score", 0.0))
    lines = [
        f"### [{title}]({url})" if url else f"### {title}",
        "",
        f"- Source: `{source}`",
        f"- Published: `{published_text}`",
        f"- Author: `{author}`",
        f"- Score: `{score:.2f}`",
        "",
        item.get("summary", "_No source summary available._"),
    ]
    llm = item.get("llm") or {}
    if llm:
        lines.extend(
            [
                "",
                "#### LLM Notes",
                f"- Relevance: `{llm.get('relevance', 0):.1f}`",
                f"- Novelty: `{llm.get('novelty', 0):.1f}`",
                f"- Impact: `{llm.get('impact', 0):.1f}`",
                f"- Editor score: `{llm.get('editor_score', 0):.1f}`",
                f"- Freshness priority: `{llm.get('freshness_priority', 0):.1f}`",
            ]
        )
        if llm.get("summary"):
            lines.append(f"- Summary: {llm['summary']}")
        if llm.get("reason"):
            lines.append(f"- Why it matters: {llm['reason']}")
    return "\n".join(lines)


def append_command_log(runtime_state: RuntimeState, message: str, max_entries: int = 12) -> None:
    timestamp = now_utc().strftime("%H:%M:%S")
    runtime_state.command_log.append(f"[{timestamp}] {message}")
    if len(runtime_state.command_log) > max_entries:
        runtime_state.command_log = runtime_state.command_log[-max_entries:]


def parse_bool_arg(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected boolean: true|false|1|0|yes|no|on|off")


def set_config_value(config: AppConfig, key: str, value: str) -> str:
    if key == "topics":
        config.topics = value
        return f"topics -> {config.topics}"
    if key == "strict_topics":
        config.strict_topics = parse_bool_arg(value)
        return f"strict_topics -> {config.strict_topics}"
    if key == "lookback_days":
        parsed = int(value)
        if parsed < 1:
            raise ValueError("lookback_days must be >= 1")
        config.lookback_days = parsed
        return f"lookback_days -> {config.lookback_days}"
    if key == "max_items":
        parsed = int(value)
        if parsed < 1:
            raise ValueError("max_items must be >= 1")
        config.max_items = parsed
        return f"max_items -> {config.max_items}"
    if key == "sort":
        parsed = value.strip().lower()
        if parsed not in {"newest", "trending"}:
            raise ValueError("sort must be newest|trending")
        config.sort_mode = parsed
        return f"sort -> {config.sort_mode}"
    if key == "quick_filter":
        config.quick_filter = value
        return f"quick_filter -> {config.quick_filter or '(empty)'}"
    if key == "refresh_minutes":
        parsed = int(value)
        if parsed < 1:
            raise ValueError("refresh_minutes must be >= 1")
        config.refresh_minutes = parsed
        return f"refresh_minutes -> {config.refresh_minutes}"
    if key == "ollama_host":
        config.ollama_host = value.strip()
        return f"ollama_host -> {config.ollama_host}"
    if key == "ollama_model":
        config.ollama_model = value.strip()
        return f"ollama_model -> {config.ollama_model}"
    if key == "analysis_limit":
        parsed = int(value)
        if parsed < 0:
            raise ValueError("analysis_limit must be >= 0")
        config.analysis_limit = parsed
        return f"analysis_limit -> {config.analysis_limit}"
    if key == "ollama_timeout_seconds":
        parsed = int(value)
        if parsed < 10:
            raise ValueError("ollama_timeout_seconds must be >= 10")
        config.ollama_timeout_seconds = parsed
        return f"ollama_timeout_seconds -> {config.ollama_timeout_seconds}"
    if key == "model_list_size":
        parsed = int(value)
        if parsed < 1:
            raise ValueError("model_list_size must be >= 1")
        config.model_list_size = parsed
        return f"model_list_size -> {config.model_list_size}"
    if key == "model_poll_seconds":
        parsed = int(value)
        if parsed < 5:
            raise ValueError("model_poll_seconds must be >= 5")
        config.model_poll_seconds = parsed
        return f"model_poll_seconds -> {config.model_poll_seconds}"
    raise ValueError(
        "unknown key. valid: topics, strict_topics, lookback_days, max_items, sort, "
        "quick_filter, refresh_minutes, ollama_host, ollama_model, analysis_limit, "
        "ollama_timeout_seconds, model_list_size, model_poll_seconds"
    )


def export_items(
    items: list[dict[str, Any]],
    llm_brief: str,
    config: AppConfig,
    fmt: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt_lower = fmt.lower()

    if fmt_lower == "json":
        payload_items: list[dict[str, Any]] = []
        for item in items:
            cloned = dict(item)
            published = item.get("published_at")
            if isinstance(published, datetime):
                cloned["published_at"] = published.isoformat()
            payload_items.append(cloned)
        payload = {
            "generated_at": now_utc().isoformat(),
            "topics": config.topics,
            "sort_mode": config.sort_mode,
            "llm_brief": llm_brief,
            "items": payload_items,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    if fmt_lower == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "source",
                    "published_at",
                    "score",
                    "author",
                    "title",
                    "url",
                    "llm_relevance",
                    "llm_novelty",
                    "llm_impact",
                    "llm_summary",
                    "summary",
                ]
            )
            for item in items:
                llm = item.get("llm") or {}
                published = item.get("published_at")
                writer.writerow(
                    [
                        item.get("source", ""),
                        published.isoformat() if isinstance(published, datetime) else "",
                        f"{item.get('score', 0.0):.4f}",
                        item.get("author", ""),
                        item.get("title", ""),
                        item.get("url", ""),
                        llm.get("relevance", ""),
                        llm.get("novelty", ""),
                        llm.get("impact", ""),
                        llm.get("summary", ""),
                        item.get("summary", ""),
                    ]
                )
        return output_path

    lines: list[str] = [
        "# AI Tracker Export",
        "",
        f"- Generated: {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Topics: {config.topics}",
        f"- Sort: {config.sort_mode}",
        "",
    ]
    if llm_brief:
        lines.extend(["## LLM Brief", llm_brief, ""])
    lines.append("## Items")
    lines.append("")
    for item in items:
        published = item.get("published_at")
        published_text = (
            published.strftime("%Y-%m-%d %H:%M UTC")
            if isinstance(published, datetime)
            else "unknown"
        )
        lines.append(
            f"- [{item.get('title', '(untitled)')}]({item.get('url', '')}) "
            f"({item.get('source', '?')}, {published_text})"
        )
        llm = item.get("llm") or {}
        if llm.get("summary"):
            lines.append(f"  - LLM: {llm['summary']}")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_config_summary(config: AppConfig) -> str:
    newsletters = ",".join(config.newsletters) if config.newsletters else "(none)"
    sources = ",".join(config.sources) if config.sources else "(none)"
    return (
        f"sources={sources} | newsletters={newsletters} | topics={config.topics} | "
        f"strict={config.strict_topics} | lookback_days={config.lookback_days} | "
        f"max_items={config.max_items} | sort={config.sort_mode} | "
        f"refresh_minutes={config.refresh_minutes} | analysis_limit={config.analysis_limit} | "
        f"ollama_model={config.ollama_model}"
    )


def handle_slash_command(
    raw_line: str,
    config: AppConfig,
    runtime_state: RuntimeState,
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> bool:
    line = raw_line.strip()
    if not line:
        return False

    if not line.startswith("/"):
        with state_lock:
            append_command_log(runtime_state, "Ignored non-command input. Use /help.")
        return False

    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        with state_lock:
            append_command_log(runtime_state, f"Command parse error: {exc}")
        return False

    if not tokens:
        return False

    command = tokens[0][1:].lower()
    args = tokens[1:]

    if command in {"help", "h", "?"}:
        with state_lock:
            runtime_state.show_menu = True
            append_command_log(
                runtime_state,
                "Commands: /help | /config | /set <key> <value> | /sources <csv> | "
                "/newsletters <all|csv> | /refresh | /export [md|json|csv] [path] | "
                "/open [index] | /brief [focus|normal|toggle|top] | /focus <section> | "
                "/size <status|brief|body> <1..12> | /layout [right 2-6] | "
                "/menu [on|off|toggle] | /clearlog | /quit",
            )
        return False

    if command == "config":
        with state_lock:
            append_command_log(runtime_state, build_config_summary(config))
        return False

    if command == "menu":
        mode = args[0].lower() if args else "toggle"
        with state_lock:
            if mode == "on":
                runtime_state.show_menu = True
                runtime_state.active_section = "menu"
            elif mode == "off":
                runtime_state.show_menu = False
                if runtime_state.active_section == "menu":
                    runtime_state.active_section = "story"
            else:
                runtime_state.show_menu = not runtime_state.show_menu
                if runtime_state.show_menu:
                    runtime_state.active_section = "menu"
            append_command_log(
                runtime_state,
                (
                    f"menu -> {'visible' if runtime_state.show_menu else 'hidden'} | "
                    f"focus -> {runtime_state.active_section}"
                ),
            )
        return False

    if command == "brief":
        mode = args[0].lower() if args else "toggle"
        with state_lock:
            if mode in {"focus", "on"}:
                runtime_state.brief_focus = True
            elif mode in {"normal", "off"}:
                runtime_state.brief_focus = False
            elif mode == "top":
                runtime_state.brief_scroll = 0
            elif mode == "toggle":
                runtime_state.brief_focus = not runtime_state.brief_focus
            else:
                append_command_log(runtime_state, "Usage: /brief [focus|normal|toggle|top]")
                return False
            append_command_log(
                runtime_state,
                f"brief_focus -> {runtime_state.brief_focus} | brief_scroll -> {runtime_state.brief_scroll}",
            )
        return False

    if command == "focus":
        if not args:
            with state_lock:
                append_command_log(
                    runtime_state,
                    f"Usage: /focus <section>; valid: {','.join(SECTION_ORDER)}",
                )
            return False
        requested = args[0].strip().lower()
        if requested not in SECTION_ORDER:
            with state_lock:
                append_command_log(
                    runtime_state,
                    f"Invalid section '{requested}'. Valid: {','.join(SECTION_ORDER)}",
                )
            return False
        with state_lock:
            runtime_state.active_section = requested
            if requested == "menu":
                runtime_state.show_menu = True
            append_command_log(runtime_state, f"focus -> {runtime_state.active_section}")
        return False

    if command == "size":
        if len(args) < 2:
            with state_lock:
                append_command_log(runtime_state, "Usage: /size <status|brief|body> <1..12>")
            return False
        row_name = args[0].strip().lower()
        try:
            parsed = clamp_row_ratio(int(args[1]))
        except ValueError:
            with state_lock:
                append_command_log(runtime_state, "Usage: /size <status|brief|body> <1..12>")
            return False
        with state_lock:
            if row_name == "status":
                runtime_state.status_row_ratio = parsed
                append_command_log(runtime_state, f"status_row_ratio -> {parsed}")
            elif row_name == "brief":
                runtime_state.brief_row_ratio = parsed
                append_command_log(runtime_state, f"brief_row_ratio -> {parsed}")
            elif row_name == "body":
                runtime_state.body_row_ratio = parsed
                append_command_log(runtime_state, f"body_row_ratio -> {parsed}")
            else:
                append_command_log(runtime_state, "Usage: /size <status|brief|body> <1..12>")
        return False

    if command == "layout":
        if len(args) >= 2 and args[0].lower() == "right":
            try:
                parsed_ratio = clamp_right_column_ratio(int(args[1]))
            except ValueError:
                with state_lock:
                    append_command_log(runtime_state, "Usage: /layout right <2..6>")
                return False
            with state_lock:
                runtime_state.right_column_ratio = parsed_ratio
                append_command_log(runtime_state, f"right_column_ratio -> {parsed_ratio}")
            return False
        with state_lock:
            append_command_log(
                runtime_state,
                "Usage: /layout right <2..6>",
            )
        return False

    if command == "clearlog":
        with state_lock:
            runtime_state.command_log.clear()
            append_command_log(runtime_state, "Command log cleared.")
        return False

    if command == "set":
        if len(args) < 2:
            with state_lock:
                append_command_log(runtime_state, "Usage: /set <key> <value>")
            return False
        key = args[0].strip().lower()
        value = " ".join(args[1:]).strip()
        try:
            with state_lock:
                message = set_config_value(config, key, value)
                append_command_log(runtime_state, message)
                if key == "refresh_minutes":
                    runtime_state.next_run_at = now_utc() + timedelta(minutes=config.refresh_minutes)
        except Exception as exc:
            with state_lock:
                append_command_log(runtime_state, f"Set failed: {exc}")
        return False

    if command == "sources":
        if not args:
            with state_lock:
                append_command_log(runtime_state, "Usage: /sources <newsletters,arxiv,x>")
            return False
        try:
            parsed_sources = parse_sources(",".join(args))
            with state_lock:
                config.sources = parsed_sources
                append_command_log(runtime_state, f"sources -> {','.join(config.sources)}")
        except Exception as exc:
            with state_lock:
                append_command_log(runtime_state, f"sources failed: {exc}")
        return False

    if command == "newsletters":
        if not args:
            with state_lock:
                append_command_log(runtime_state, "Usage: /newsletters <all|csv>")
            return False
        try:
            parsed_newsletters = parse_newsletters(",".join(args))
            with state_lock:
                config.newsletters = parsed_newsletters
                append_command_log(
                    runtime_state,
                    f"newsletters -> {','.join(config.newsletters) if config.newsletters else '(none)'}",
                )
        except Exception as exc:
            with state_lock:
                append_command_log(runtime_state, f"newsletters failed: {exc}")
        return False

    if command == "refresh":
        with state_lock:
            runtime_state.next_run_at = now_utc()
            append_command_log(runtime_state, "Manual refresh requested.")
        return False

    if command == "open":
        with state_lock:
            items_snapshot = runtime_state.cycle_state.get("items", [])
            selected = runtime_state.selected_index
        if not items_snapshot:
            with state_lock:
                append_command_log(runtime_state, "No stories available to open.")
            return False

        if args:
            try:
                requested = int(args[0]) - 1
            except ValueError:
                with state_lock:
                    append_command_log(runtime_state, "Usage: /open [1-based-index]")
                return False
        else:
            requested = selected
        requested = clamp_selection(requested, items_snapshot)
        target = items_snapshot[requested]
        error = open_link(target.get("url", ""))
        with state_lock:
            runtime_state.selected_index = requested
            if error:
                append_command_log(runtime_state, error)
            else:
                append_command_log(runtime_state, f"Opened story {requested + 1}: {target.get('title', '(untitled)')}")
        return False

    if command == "export":
        fmt = "md"
        path_arg: str | None = None
        if args:
            first = args[0].lower()
            if first in {"md", "json", "csv"}:
                fmt = first
                if len(args) > 1:
                    path_arg = args[1]
            else:
                path_arg = args[0]
        if path_arg:
            output_path = Path(path_arg).expanduser()
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{fmt}")
            fmt = output_path.suffix.lstrip(".").lower() or fmt
        else:
            stamp = now_utc().strftime("%Y%m%d-%H%M%S")
            output_path = Path.cwd() / "exports" / f"ai-feed-{stamp}.{fmt}"

        with state_lock:
            items_snapshot = copy.deepcopy(runtime_state.cycle_state.get("items", []))
            llm_brief = runtime_state.cycle_state.get("llm_brief", "")
            config_snapshot = copy.deepcopy(config)
        try:
            exported = export_items(items_snapshot, llm_brief, config_snapshot, fmt, output_path)
            with state_lock:
                append_command_log(runtime_state, f"Exported {len(items_snapshot)} items -> {exported}")
        except Exception as exc:
            with state_lock:
                append_command_log(runtime_state, f"Export failed: {exc}")
        return False

    if command in {"quit", "exit"}:
        with state_lock:
            append_command_log(runtime_state, "Quit requested.")
        stop_event.set()
        return True

    with state_lock:
        append_command_log(runtime_state, f"Unknown command: {line}. Use /help.")
    return False


def refresh_worker(
    config: AppConfig,
    runtime_state: RuntimeState,
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        with state_lock:
            wait_seconds = (runtime_state.next_run_at - now_utc()).total_seconds()

        if wait_seconds > 0:
            stop_event.wait(min(wait_seconds, 1.0))
            continue

        with state_lock:
            runtime_state.is_refreshing = True
            runtime_state.status_message = "Refreshing feeds and running analysis..."
            config_snapshot = copy.deepcopy(config)

        cycle_state = run_cycle(config_snapshot)

        with state_lock:
            runtime_state.cycle_state = cycle_state
            runtime_state.selected_index = clamp_selection(
                runtime_state.selected_index,
                runtime_state.cycle_state.get("items", []),
            )
            runtime_state.brief_scroll = 0
            runtime_state.next_run_at = now_utc() + timedelta(minutes=config.refresh_minutes)
            runtime_state.is_refreshing = False
            runtime_state.status_message = "Idle"


def render_models_table(models: list[dict[str, Any]], selected_model: str, limit: int) -> Table:
    table = Table(title="Ollama Models (live, size-sorted)", expand=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", no_wrap=True, overflow="ellipsis")
    table.add_column("Size", justify="right", width=10)
    table.add_column("Age", justify="right", width=6)

    shown = models[: max(limit, 1)]
    for idx, model in enumerate(shown, start=1):
        marker = "*" if model["name"] == selected_model else ""
        modified = parse_date(model.get("modified_at"))
        modified_display = human_age(modified) if modified else "-"
        table.add_row(
            str(idx),
            f"{model['name']} {marker}".strip(),
            humanize_bytes(model.get("size_bytes")),
            modified_display,
        )

    if not shown:
        table.add_row("-", "No models discovered", "-", "-")
    return table


def render_feed_table(
    items: list[dict[str, Any]],
    max_rows: int,
    empty_message: str,
    selected_index: int,
) -> Table:
    table = Table(title="AI Feed", expand=True)
    table.add_column("Sel", width=3)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Src", width=11)
    table.add_column("Age", justify="right", width=6)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Title")
    table.add_column("LLM", width=20)

    for idx, item in enumerate(items[:max_rows], start=1):
        llm = item.get("llm") or {}
        llm_summary = "-"
        if llm:
            llm_summary = (
                f"R{llm.get('relevance', 0):.1f} N{llm.get('novelty', 0):.1f} "
                f"I{llm.get('impact', 0):.1f}"
            )
        is_selected = idx - 1 == selected_index
        marker = ">" if is_selected else ""
        title_text = truncate(item["title"], 92)
        if item.get("url"):
            title_cell = f"[link={item['url']}]{title_text}[/link]"
        else:
            title_cell = title_text
        table.add_row(
            marker,
            str(idx),
            item["source"],
            human_age(item["published_at"]),
            f"{item['score']:.2f}",
            title_cell,
            llm_summary,
            style="bold bright_white on rgb(28,28,28)" if is_selected else "",
        )

    if not items:
        table.add_row("", "-", "-", "-", "-", empty_message, "-")
    return table


def render_source_counts(items: list[dict[str, Any]]) -> Table:
    counts = Counter(item["source"] for item in items)
    table = Table(title="Source Mix", expand=True)
    table.add_column("Source")
    table.add_column("Items", justify="right")
    if not counts:
        table.add_row("-", "0")
        return table
    for source, count in sorted(counts.items(), key=lambda kv: kv[0]):
        table.add_row(source, str(count))
    return table


def render_selected_story_panel(items: list[dict[str, Any]], selected_index: int) -> Panel:
    if not items:
        return Panel("_No story selected yet._", title="Selected Story", border_style="cyan")
    index = clamp_selection(selected_index, items)
    item = items[index]
    markdown = Markdown(format_story_markdown(item))
    return Panel(markdown, title=f"Selected Story ({index + 1}/{len(items)})", border_style="cyan")


def render_command_panel(
    command_log: list[str],
    in_command_mode: bool,
    command_buffer: str,
) -> Panel:
    if not command_log:
        body = "Type /help then press Enter."
    else:
        body = "\n".join(command_log[-10:])
    mode_label = "COMMAND MODE" if in_command_mode else "NORMAL MODE"
    buffer_line = command_buffer if in_command_mode else "Press / to start command input."
    merged = f"{mode_label}\n{buffer_line}\n\n{body}"
    return Panel(merged, title="Slash Commands", border_style="magenta")


def render_menu_panel(show_menu: bool) -> Panel:
    if not show_menu:
        return Panel("Press [m] for menu", title="Menu", border_style="blue")
    body = (
        "Hotkeys:\n"
        "- / : enter command mode\n"
        "- Enter : submit command (or open story in normal mode)\n"
        "- Tab / Shift+Tab : cycle focused section\n"
        "- j / k : next/prev story\n"
        "- Up / Down : story nav (or brief scroll when brief focused)\n"
        "- PgUp / PgDn : scroll brief\n"
        "- + / - : grow/shrink focused row\n"
        "- o : open selected story link\n"
        "- b : toggle brief focus mode\n"
        "- [ / ] : resize right column narrower/wider\n"
        "- Esc : cancel command mode\n"
        "- q : quit\n"
        "- m : toggle this menu\n\n"
        "Slash commands:\n"
        "- /help\n"
        "- /config\n"
        "- /set <key> <value>\n"
        "- /sources <newsletters,arxiv,x>\n"
        "- /newsletters <all|csv>\n"
        "- /refresh\n"
        "- /open [index]\n"
        "- /brief [focus|normal|toggle|top]\n"
        "- /focus <section>\n"
        "- /size <status|brief|body> <1..12>\n"
        "- /layout right <2..6>\n"
        "- /export [md|json|csv] [path]\n"
        "- /menu [on|off|toggle]\n"
        "- /clearlog\n"
        "- /quit"
    )
    return Panel(body, title="Menu", border_style="blue")


def render_warning_bar(warnings: list[str], terminal_width: int) -> Text:
    if not warnings:
        return Text("Warnings: none", style="yellow")
    clean = [normalize_text(warn) for warn in warnings if normalize_text(warn)]
    if not clean:
        clean = ["Unknown warning"]
    shown = clean[:2]
    text = " | ".join(shown)
    if len(clean) > 2:
        text += f" | (+{len(clean) - 2} more)"
    max_chars = max(40, terminal_width - 24)
    text = truncate(text, max_chars)
    return Text(f"Warnings: {text}", style="yellow")


def render_right_section(
    active_section: str,
    items: list[dict[str, Any]],
    selected_index: int,
    models: list[dict[str, Any]],
    config: AppConfig,
    show_menu: bool,
    command_log: list[str],
    in_command_mode: bool,
    command_buffer: str,
) -> tuple[str, Any]:
    if active_section == "models":
        return "Models", render_models_table(models, config.ollama_model, config.model_list_size)
    if active_section == "sources":
        return "Source Mix", render_source_counts(items)
    if active_section == "menu":
        menu_panel = render_menu_panel(True)
        return str(menu_panel.title or "Menu"), menu_panel.renderable
    if active_section == "commands":
        command_panel = render_command_panel(command_log, in_command_mode, command_buffer)
        return str(command_panel.title or "Commands"), command_panel.renderable
    if show_menu:
        menu_panel = render_menu_panel(True)
        return str(menu_panel.title or "Menu"), menu_panel.renderable
    story_panel = render_selected_story_panel(items, selected_index)
    return str(story_panel.title or "Selected Story"), story_panel.renderable


def build_dashboard(
    config: AppConfig,
    cycle_state: dict[str, Any],
    seconds_to_next_run: int,
    is_refreshing: bool,
    status_message: str,
    command_log: list[str],
    show_menu: bool,
    in_command_mode: bool,
    command_buffer: str,
    selected_index: int,
    right_column_ratio: int,
    brief_focus: bool,
    brief_scroll: int,
    active_section: str,
    status_row_ratio: int,
    brief_row_ratio: int,
    body_row_ratio: int,
    terminal_width: int,
    terminal_height: int,
) -> Panel:
    started: datetime = cycle_state["started"]
    elapsed_seconds: float = cycle_state["elapsed_seconds"]
    items: list[dict[str, Any]] = cycle_state["items"]
    errors: list[str] = cycle_state["errors"]
    llm_brief: str = cycle_state["llm_brief"]
    models: list[dict[str, Any]] = cycle_state["models"]
    model_error: str = cycle_state["model_error"]

    brief_panel, max_brief_scroll, clamped_brief_scroll = build_brief_panel(
        llm_brief=llm_brief,
        brief_focus=brief_focus,
        brief_scroll=brief_scroll,
        terminal_width=terminal_width,
        terminal_height=terminal_height,
        focused=(active_section == "brief"),
    )

    topics_short = truncate(config.topics or "(none)", 64)
    status_line_1 = (
        f"Refresh: {started.strftime('%Y-%m-%d %H:%M:%S UTC')} | Next: {seconds_to_next_run}s | "
        f"Cycle: {elapsed_seconds:.1f}s | State: {'refreshing' if is_refreshing else 'idle'} | "
        f"Focus: {active_section} | Mode: {'strict' if config.strict_topics else 'broad'} | "
        f"Sort: {config.sort_mode}"
    )
    status_line_2 = (
        f"Topics: {topics_short} | Rows={status_row_ratio}:{brief_row_ratio}:{body_row_ratio} | "
        f"Right={right_column_ratio} | Brief={clamped_brief_scroll}/{max_brief_scroll} | "
        f"Model={config.ollama_model} | Limit={config.analysis_limit} | "
        "Hotkeys: Tab section, j/k stories, PgUp/PgDn brief, +/- row size, [ ] width, q quit"
    )
    status_lines = [
        truncate(status_line_1, max(60, terminal_width - 12)),
        truncate(status_line_2, max(60, terminal_width - 12)),
    ]
    status_panel = Panel(
        "\n".join(status_lines),
        title="Status",
        border_style="bright_cyan" if active_section == "status" else "cyan",
    )

    warnings: list[str] = []
    if model_error:
        warnings.append(model_error)
    warnings.extend(errors)
    warning_bar = render_warning_bar(warnings, terminal_width)

    if brief_focus:
        focus_layout = Layout(name="root")
        focus_layout.split_column(
            Layout(status_panel, name="top", ratio=clamp_row_ratio(status_row_ratio)),
            Layout(
                brief_panel,
                name="brief",
                ratio=max(clamp_row_ratio(brief_row_ratio), clamp_row_ratio(body_row_ratio)),
            ),
            Layout(warning_bar, name="warnings", size=1),
        )
        return Panel(
            focus_layout,
            title="AI Tracker Terminal",
            border_style="bright_blue",
        )

    empty_message = (
        "Loading first refresh... TUI is active and updates will appear shortly."
        if is_refreshing
        else "No items for current filters"
    )
    feed_table = render_feed_table(items, config.max_items, empty_message, selected_index)
    feed_panel = Panel(
        feed_table,
        title="Feed",
        border_style="bright_cyan" if active_section == "feed" else "cyan",
        padding=(0, 0),
    )
    right_title, right_content = render_right_section(
        active_section=active_section,
        items=items,
        selected_index=selected_index,
        models=models,
        config=config,
        show_menu=show_menu,
        command_log=command_log,
        in_command_mode=in_command_mode,
        command_buffer=command_buffer,
    )
    right_panel = Panel(
        right_content,
        title=right_title,
        border_style="bright_cyan" if active_section in RIGHT_SECTIONS else "cyan",
        padding=(0, 0),
    )
    left_ratio = max(2, 8 - clamp_right_column_ratio(right_column_ratio))
    right_ratio = clamp_right_column_ratio(right_column_ratio)
    bottom_layout = Layout(name="bottom")
    bottom_layout.split_row(
        Layout(feed_panel, name="feed", ratio=left_ratio),
        Layout(right_panel, name="right", ratio=right_ratio),
    )

    root_layout = Layout(name="root")
    root_layout.split_column(
        Layout(status_panel, name="top", ratio=clamp_row_ratio(status_row_ratio)),
        Layout(brief_panel, name="brief", ratio=clamp_row_ratio(brief_row_ratio)),
        Layout(bottom_layout, name="bottom", ratio=clamp_row_ratio(body_row_ratio)),
        Layout(warning_bar, name="warnings", size=1),
    )
    return Panel(
        root_layout,
        title="AI Tracker Terminal",
        border_style="bright_blue",
    )


def parse_args(argv: list[str]) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="AI Tracker terminal app with hourly refresh and Ollama analysis."
    )
    parser.add_argument("--topics", default=DEFAULT_TOPICS)
    parser.add_argument(
        "--sources",
        default="newsletters,arxiv,x",
        help="Comma-separated: newsletters,arxiv,x",
    )
    parser.add_argument("--strict-topics", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=15)
    parser.add_argument("--max-items", type=int, default=25)
    parser.add_argument("--sort", choices=["newest", "trending"], default="trending")
    parser.add_argument("--quick-filter", default="")
    parser.add_argument(
        "--newsletters",
        default="all",
        help="Comma-separated newsletter names (case-insensitive) or 'all'.",
    )
    parser.add_argument("--refresh-minutes", type=int, default=60)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--analysis-limit", type=int, default=15)
    parser.add_argument("--ollama-timeout-seconds", type=int, default=600)
    parser.add_argument("--model-list-size", type=int, default=8)
    parser.add_argument("--model-poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")

    args = parser.parse_args(argv)

    if args.lookback_days < 1:
        raise ValueError("--lookback-days must be >= 1")
    if args.max_items < 1:
        raise ValueError("--max-items must be >= 1")
    if args.refresh_minutes < 1:
        raise ValueError("--refresh-minutes must be >= 1")
    if args.analysis_limit < 0:
        raise ValueError("--analysis-limit must be >= 0")
    if args.ollama_timeout_seconds < 10:
        raise ValueError("--ollama-timeout-seconds must be >= 10")
    if args.model_list_size < 1:
        raise ValueError("--model-list-size must be >= 1")
    if args.model_poll_seconds < 5:
        raise ValueError("--model-poll-seconds must be >= 5")

    sources = parse_sources(args.sources)
    newsletters = parse_newsletters(args.newsletters)

    return AppConfig(
        topics=args.topics,
        sources=sources,
        strict_topics=args.strict_topics,
        lookback_days=args.lookback_days,
        max_items=args.max_items,
        sort_mode=args.sort,
        newsletters=newsletters,
        quick_filter=args.quick_filter,
        refresh_minutes=args.refresh_minutes,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        analysis_limit=args.analysis_limit,
        ollama_timeout_seconds=args.ollama_timeout_seconds,
        model_list_size=args.model_list_size,
        model_poll_seconds=args.model_poll_seconds,
        once=args.once,
    )


def _line_input_worker(
    command_queue: queue.Queue[tuple[str, str]],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            if stop_event.wait(0.2):
                break
            continue
        if line == "":
            if stop_event.wait(0.2):
                break
            continue
        command_queue.put(("command_line", line.rstrip("\n")))


def command_input_worker(
    command_queue: queue.Queue[tuple[str, str]],
    stop_event: threading.Event,
) -> None:
    if not sys.stdin.isatty():
        _line_input_worker(command_queue, stop_event)
        return

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except Exception:
        _line_input_worker(command_queue, stop_event)
        return

    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            data = os.read(fd, 1)
            if not data:
                continue
            key = data.decode("utf-8", errors="ignore")
            if not key:
                continue
            if key in {"\r", "\n"}:
                command_queue.put(("key", "ENTER"))
                continue
            if key == "\t":
                command_queue.put(("key", "TAB"))
                continue
            if key in {"\x7f", "\b"}:
                command_queue.put(("key", "BACKSPACE"))
                continue
            if key == "\x1b":
                sequence = ""
                while select.select([fd], [], [], 0.001)[0]:
                    sequence += os.read(fd, 1).decode("utf-8", errors="ignore")
                    if not sequence:
                        continue
                    if sequence[-1].isalpha() or sequence.endswith("~") or len(sequence) >= 6:
                        break
                if sequence == "[A":
                    command_queue.put(("key", "UP"))
                elif sequence == "[B":
                    command_queue.put(("key", "DOWN"))
                elif sequence == "[Z":
                    command_queue.put(("key", "SHTAB"))
                elif sequence == "[5~":
                    command_queue.put(("key", "PGUP"))
                elif sequence == "[6~":
                    command_queue.put(("key", "PGDN"))
                else:
                    command_queue.put(("key", "ESC"))
                continue
            if key == "\x03":
                command_queue.put(("key", "QUIT"))
                continue
            command_queue.put(("key", key))
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def run(config: AppConfig, console: Console) -> int:
    if config.once:
        cycle_state = run_cycle(config)
        console.print(
            build_dashboard(
                config,
                cycle_state,
                seconds_to_next_run=0,
                is_refreshing=False,
                status_message="Completed one-shot run.",
                command_log=[],
                show_menu=False,
                in_command_mode=False,
                command_buffer="",
                selected_index=0,
                right_column_ratio=3,
                brief_focus=False,
                brief_scroll=0,
                active_section="feed",
                status_row_ratio=4,
                brief_row_ratio=5,
                body_row_ratio=9,
                terminal_width=console.size.width,
                terminal_height=console.size.height,
            )
        )
        return 0

    runtime_state = RuntimeState(
        cycle_state=make_initial_cycle_state(config),
        next_run_at=now_utc(),
        is_refreshing=True,
        status_message="Starting first refresh...",
        command_log=[],
        selected_index=0,
        right_column_ratio=3,
        brief_focus=False,
        brief_scroll=0,
        active_section="feed",
        status_row_ratio=4,
        brief_row_ratio=5,
        body_row_ratio=9,
        show_menu=False,
        in_command_mode=False,
        command_buffer="",
    )
    append_command_log(
        runtime_state,
        "Press / for commands. Tab cycles sections. +/- resizes rows. q quits.",
    )
    state_lock = threading.Lock()
    stop_event = threading.Event()
    command_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    worker = threading.Thread(
        target=refresh_worker,
        args=(config, runtime_state, state_lock, stop_event),
        daemon=True,
    )
    worker.start()
    command_worker = threading.Thread(
        target=command_input_worker,
        args=(command_queue, stop_event),
        daemon=True,
    )
    command_worker.start()

    last_model_poll_at = now_utc()
    initial_snapshot = copy.deepcopy(runtime_state.cycle_state)
    initial_command_log = list(runtime_state.command_log)
    initial_show_menu = runtime_state.show_menu
    initial_in_command_mode = runtime_state.in_command_mode
    initial_command_buffer = runtime_state.command_buffer
    initial_selected_index = runtime_state.selected_index
    initial_right_column_ratio = runtime_state.right_column_ratio
    initial_brief_focus = runtime_state.brief_focus
    initial_brief_scroll = runtime_state.brief_scroll
    initial_active_section = runtime_state.active_section
    initial_status_row_ratio = runtime_state.status_row_ratio
    initial_brief_row_ratio = runtime_state.brief_row_ratio
    initial_body_row_ratio = runtime_state.body_row_ratio
    with Live(
        build_dashboard(
            config,
            initial_snapshot,
            seconds_to_next_run=0,
            is_refreshing=True,
            status_message="Starting first refresh...",
            command_log=initial_command_log,
            show_menu=initial_show_menu,
            in_command_mode=initial_in_command_mode,
            command_buffer=initial_command_buffer,
            selected_index=initial_selected_index,
            right_column_ratio=initial_right_column_ratio,
            brief_focus=initial_brief_focus,
            brief_scroll=initial_brief_scroll,
            active_section=initial_active_section,
            status_row_ratio=initial_status_row_ratio,
            brief_row_ratio=initial_brief_row_ratio,
            body_row_ratio=initial_body_row_ratio,
            terminal_width=console.size.width,
            terminal_height=console.size.height,
        ),
        console=console,
        refresh_per_second=4,
        screen=True,
    ) as live:
        try:
            while True:
                exit_requested = False
                while True:
                    try:
                        event_type, event_value = command_queue.get_nowait()
                    except queue.Empty:
                        break
                    command_to_submit = ""
                    open_requested = False
                    open_url = ""
                    open_title = ""
                    if event_type == "command_line":
                        command_to_submit = event_value
                    elif event_type == "key":
                        if event_value == "QUIT":
                            exit_requested = True
                            break
                        with state_lock:
                            in_command_mode = runtime_state.in_command_mode
                            if not in_command_mode:
                                lowered = event_value.lower() if len(event_value) == 1 else ""
                                items = runtime_state.cycle_state.get("items", [])
                                if lowered == "q":
                                    append_command_log(runtime_state, "Quit requested by hotkey.")
                                    exit_requested = True
                                    break
                                if lowered == "m":
                                    runtime_state.show_menu = not runtime_state.show_menu
                                    if runtime_state.show_menu:
                                        runtime_state.active_section = "menu"
                                    append_command_log(
                                        runtime_state,
                                        (
                                            f"menu -> {'visible' if runtime_state.show_menu else 'hidden'} | "
                                            f"focus -> {runtime_state.active_section}"
                                        ),
                                    )
                                    continue
                                if lowered == "b":
                                    runtime_state.brief_focus = not runtime_state.brief_focus
                                    if runtime_state.brief_focus:
                                        runtime_state.active_section = "brief"
                                    append_command_log(
                                        runtime_state,
                                        (
                                            f"brief_focus -> {runtime_state.brief_focus} | "
                                            f"focus -> {runtime_state.active_section}"
                                        ),
                                    )
                                    continue
                                if event_value == "TAB":
                                    runtime_state.active_section = cycle_section(
                                        runtime_state.active_section,
                                        1,
                                    )
                                    continue
                                if event_value == "SHTAB":
                                    runtime_state.active_section = cycle_section(
                                        runtime_state.active_section,
                                        -1,
                                    )
                                    continue
                                if event_value == "PGUP":
                                    runtime_state.brief_scroll = max(0, runtime_state.brief_scroll - 8)
                                    continue
                                if event_value == "PGDN":
                                    runtime_state.brief_scroll += 8
                                    continue
                                if event_value in {"+", "="}:
                                    message = adjust_focused_row_height(runtime_state, 1)
                                    append_command_log(runtime_state, message)
                                    continue
                                if event_value in {"-", "_"}:
                                    message = adjust_focused_row_height(runtime_state, -1)
                                    append_command_log(runtime_state, message)
                                    continue
                                if event_value == "[":
                                    runtime_state.right_column_ratio = clamp_right_column_ratio(
                                        runtime_state.right_column_ratio - 1
                                    )
                                    append_command_log(
                                        runtime_state,
                                        f"right_column_ratio -> {runtime_state.right_column_ratio}",
                                    )
                                    continue
                                if event_value == "]":
                                    runtime_state.right_column_ratio = clamp_right_column_ratio(
                                        runtime_state.right_column_ratio + 1
                                    )
                                    append_command_log(
                                        runtime_state,
                                        f"right_column_ratio -> {runtime_state.right_column_ratio}",
                                    )
                                    continue
                                if event_value == "/":
                                    runtime_state.in_command_mode = True
                                    runtime_state.command_buffer = "/"
                                    runtime_state.status_message = "Command mode active."
                                    continue
                                if event_value == "UP":
                                    if runtime_state.active_section == "brief":
                                        runtime_state.brief_scroll = max(0, runtime_state.brief_scroll - 2)
                                    else:
                                        runtime_state.selected_index = cycle_selection(
                                            runtime_state.selected_index,
                                            items,
                                            -1,
                                        )
                                    continue
                                if event_value == "DOWN":
                                    if runtime_state.active_section == "brief":
                                        runtime_state.brief_scroll += 2
                                    else:
                                        runtime_state.selected_index = cycle_selection(
                                            runtime_state.selected_index,
                                            items,
                                            1,
                                        )
                                    continue
                                if lowered == "j":
                                    runtime_state.selected_index = cycle_selection(
                                        runtime_state.selected_index,
                                        items,
                                        1,
                                    )
                                    continue
                                if lowered == "k":
                                    runtime_state.selected_index = cycle_selection(
                                        runtime_state.selected_index,
                                        items,
                                        -1,
                                    )
                                    continue
                                if event_value == "ENTER" or lowered == "o":
                                    if items:
                                        selected = clamp_selection(runtime_state.selected_index, items)
                                        runtime_state.selected_index = selected
                                        target = items[selected]
                                        open_requested = True
                                        open_url = target.get("url", "")
                                        open_title = target.get("title", "(untitled)")
                                    else:
                                        append_command_log(runtime_state, "No stories available to open.")
                                    continue
                                continue

                            if event_value == "ESC":
                                runtime_state.in_command_mode = False
                                runtime_state.command_buffer = ""
                                runtime_state.status_message = "Command mode cancelled."
                                append_command_log(runtime_state, "Command input cancelled.")
                                continue
                            if event_value == "ENTER":
                                command_to_submit = runtime_state.command_buffer.strip()
                                runtime_state.in_command_mode = False
                                runtime_state.command_buffer = ""
                                runtime_state.status_message = "Processing command..."
                            elif event_value == "BACKSPACE":
                                runtime_state.command_buffer = runtime_state.command_buffer[:-1]
                            elif len(event_value) == 1 and event_value.isprintable():
                                runtime_state.command_buffer += event_value
                    if open_requested:
                        error = open_link(open_url)
                        with state_lock:
                            if error:
                                append_command_log(runtime_state, error)
                            else:
                                append_command_log(runtime_state, f"Opened: {open_title}")
                    if command_to_submit:
                        if handle_slash_command(
                            command_to_submit,
                            config,
                            runtime_state,
                            state_lock,
                            stop_event,
                        ):
                            exit_requested = True
                            break
                if exit_requested:
                    return 0

                now = now_utc()
                should_poll_models = (
                    now - last_model_poll_at
                ).total_seconds() >= config.model_poll_seconds
                if should_poll_models:
                    models_result = fetch_ollama_models(config.ollama_host)
                    with state_lock:
                        runtime_state.cycle_state["models"] = models_result["models"]
                        runtime_state.cycle_state["model_error"] = models_result["error"]
                    last_model_poll_at = now

                with state_lock:
                    snapshot = copy.deepcopy(runtime_state.cycle_state)
                    next_run_at = runtime_state.next_run_at
                    is_refreshing = runtime_state.is_refreshing
                    status_message = runtime_state.status_message
                    command_log = list(runtime_state.command_log)
                    selected_index = runtime_state.selected_index
                    right_column_ratio = runtime_state.right_column_ratio
                    brief_focus = runtime_state.brief_focus
                    brief_scroll = runtime_state.brief_scroll
                    active_section = runtime_state.active_section
                    status_row_ratio = runtime_state.status_row_ratio
                    brief_row_ratio = runtime_state.brief_row_ratio
                    body_row_ratio = runtime_state.body_row_ratio
                    show_menu = runtime_state.show_menu
                    in_command_mode = runtime_state.in_command_mode
                    command_buffer = runtime_state.command_buffer

                seconds_to_next = max(0, int((next_run_at - now).total_seconds()))
                live.update(
                    build_dashboard(
                        config,
                        snapshot,
                        seconds_to_next,
                        is_refreshing=is_refreshing,
                        status_message=status_message,
                        command_log=command_log,
                        selected_index=selected_index,
                        show_menu=show_menu,
                        in_command_mode=in_command_mode,
                        command_buffer=command_buffer,
                        right_column_ratio=right_column_ratio,
                        brief_focus=brief_focus,
                        brief_scroll=brief_scroll,
                        active_section=active_section,
                        status_row_ratio=status_row_ratio,
                        brief_row_ratio=brief_row_ratio,
                        body_row_ratio=body_row_ratio,
                        terminal_width=console.size.width,
                        terminal_height=console.size.height,
                    )
                )
                time.sleep(1)
        finally:
            stop_event.set()
            worker.join(timeout=2)
            command_worker.join(timeout=2)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    console = Console()
    try:
        config = parse_args(argv if argv is not None else sys.argv[1:])
    except Exception as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2

    try:
        return run(config, console)
    except KeyboardInterrupt:
        console.print("\n[bold]Stopped.[/bold]")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
