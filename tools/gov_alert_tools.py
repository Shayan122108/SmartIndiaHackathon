"""
Government Outbreak Alert Tools
================================
Polls publicly accessible official sources for disease outbreak and disaster alerts
relevant to India / Telangana, then converts them to the project's internal alert format.

Sources used (no API keys required):
  1. WHO Disease Outbreak News RSS  — https://www.who.int/rss-feeds/news-en.xml
  2. NDMA Disaster Alerts RSS       — https://ndma.gov.in/Media/DisasterAlerts (RSS)
  3. ProMED International RSS       — https://promedmail.org/promed-rss/ (fallback)
"""

import logging
import hashlib
import re
import time as _time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logging.warning("feedparser not installed — gov alert polling disabled. Run: pip install feedparser")

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS Feed URLs
# ---------------------------------------------------------------------------

FEEDS = {
    # ECDC Epidemiological Updates — confirmed working RSS from ecdc.europa.eu/en/rss-feeds
    "ECDC Epidemiological Updates": {
        "url": "https://www.ecdc.europa.eu/en/taxonomy/term/1310/feed",
        "fallback_url": None,
        "source_tag": "ECDC",
        "india_keywords": [
            "india", "south asia", "asia", "dengue", "cholera", "typhoid",
            "malaria", "leptospirosis", "nipah", "rabies", "encephalitis",
            "h5n1", "h1n1", "avian influenza", "outbreak", "epidemic",
        ],
        "global_emergency_keywords": [
            "pandemic", "pheic", "public health emergency", "novel",
            "unknown", "hemorrhagic", "plague", "global", "worldwide",
        ],
    },
    # ECDC Risk Assessments — rapid risk assessments for emerging threats
    "ECDC Risk Assessments": {
        "url": "https://www.ecdc.europa.eu/en/taxonomy/term/1295/feed",
        "fallback_url": None,
        "source_tag": "ECDC",
        "india_keywords": [
            "india", "asia", "dengue", "cholera", "malaria", "outbreak",
            "typhoid", "influenza", "avian", "nipah",
        ],
        "global_emergency_keywords": [
            "pandemic", "novel", "pheic", "risk assessment", "threat",
            "hemorrhagic", "emerging", "unknown",
        ],
    },
    # ECDC Avian Influenza — tracked specifically given H5N1 spread risk to South Asia
    "ECDC Avian Influenza": {
        "url": "https://www.ecdc.europa.eu/en/taxonomy/term/323//feed",
        "fallback_url": None,
        "source_tag": "ECDC",
        "india_keywords": [
            "india", "asia", "south asia", "poultry", "h5n1", "h5n2", "h7n9",
            "avian", "bird flu", "influenza",
        ],
        "global_emergency_keywords": [
            "pandemic", "human case", "human infection", "novel", "zoonotic",
        ],
    },
    # ECDC Mpox feed
    "ECDC Mpox": {
        "url": "https://www.ecdc.europa.eu/en/taxonomy/term/2794/feed",
        "fallback_url": None,
        "source_tag": "ECDC",
        "india_keywords": [
            "india", "asia", "south asia", "mpox", "monkeypox", "outbreak",
        ],
        "global_emergency_keywords": [
            "pandemic", "pheic", "global", "spread", "novel clade",
        ],
    },
    # ECDC News/Press Releases — broader coverage
    "ECDC News": {
        "url": "https://www.ecdc.europa.eu/en/taxonomy/term/1307/feed",
        "fallback_url": None,
        "source_tag": "ECDC",
        "india_keywords": [
            "india", "asia", "dengue", "cholera", "malaria", "typhoid",
            "nipah", "outbreak", "epidemic",
        ],
        "global_emergency_keywords": [
            "pandemic", "pheic", "emergency", "alert", "novel pathogen",
        ],
    },
}

# ---------------------------------------------------------------------------
# Internal Alert Level Mapping
# ---------------------------------------------------------------------------

def _infer_alert_level(title: str, summary: str) -> str:
    combined = (title + " " + summary).lower()
    if any(w in combined for w in ["pandemic", "pheic", "critical", "emergency", "evacuate", "deaths"]):
        return "CRITICAL"
    if any(w in combined for w in ["outbreak", "flood", "cyclone", "high risk", "urgent"]):
        return "HIGH"
    if any(w in combined for w in ["warning", "alert", "cases", "reported", "risk"]):
        return "MODERATE"
    return "LOW"


def _infer_alert_type(source_tag: str, title: str) -> str:
    title_lower = title.lower()
    if source_tag == "NDMA":
        return "disaster_alert"
    if any(w in title_lower for w in ["outbreak", "epidemic", "disease", "virus", "fever", "cholera", "dengue"]):
        return "epidemic_warning"
    return "government_advisory"


def _stable_id(url: str, title: str) -> str:
    """Generate a stable deduplication key from URL + title."""
    raw = f"{url}::{title}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _parse_published(entry) -> str:
    """Extract and normalise the published timestamp from a feedparser entry."""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            import time
            return datetime.fromtimestamp(
                time.mktime(entry.published_parsed), tz=timezone.utc
            ).isoformat()
    except Exception:
        pass
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core Entry → Alert Converter
# ---------------------------------------------------------------------------

def parse_rss_entry_to_alert(entry, source_name: str, source_tag: str) -> Dict[str, Any]:
    """Convert a single feedparser entry to our internal alert dict format."""
    title = getattr(entry, "title", "Government Health Alert")
    link = getattr(entry, "link", "")
    summary_raw = getattr(entry, "summary", getattr(entry, "description", ""))

    # Strip HTML tags naively (feedparser usually handles this, but be safe)
    import re
    summary = re.sub(r"<[^>]+>", " ", summary_raw).strip()
    summary = " ".join(summary.split())[:500]  # Trim to 500 chars

    published = _parse_published(entry)
    level = _infer_alert_level(title, summary)
    alert_type = _infer_alert_type(source_tag, title)

    return {
        "alert_id": _stable_id(link, title),
        "source": source_name,
        "source_tag": source_tag,
        "alert_type": alert_type,
        "alert_level": level,
        "title": title,
        "summary": summary[:300] if summary else title,
        "link": link,
        "details": {
            "full_summary": summary,
            "source_url": link,
            "source_name": source_name,
        },
        "recommendations": [
            "Monitor official health advisories",
            "Follow local health department guidelines",
            "Report unusual symptoms to district health office: 1075",
        ],
        "emergency_contacts": {
            "health_department": "1075",
            "ambulance": "108",
            "disaster_management": "1070",
        },
        "timestamp": published,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "is_government_source": True,
    }


# ---------------------------------------------------------------------------
# Fetch with User-Agent header (many feeds block default feedparser UA)
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HealthAlertBot/1.0; +https://warangal-health.gov.in)"
}


def _fetch_raw(url: str) -> Optional[str]:
    """Fetch raw feed content via requests with proper headers."""
    if not REQUESTS_AVAILABLE:
        return None
    try:
        resp = _requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"requests fetch failed for {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-Feed Fetcher
# ---------------------------------------------------------------------------

def _fetch_feed(
    feed_name: str,
    feed_config: Dict[str, Any],
    india_only: bool = False,
    max_entries: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch and filter a single RSS feed.
    Tries the primary URL first, then fallback_url if specified.
    Uses requests to fetch raw XML so we can pass a browser User-Agent.
    """
    if not FEEDPARSER_AVAILABLE:
        return []

    source_tag = feed_config["source_tag"]
    india_kw = [k.lower() for k in feed_config.get("india_keywords", [])]
    global_kw = [k.lower() for k in feed_config.get("global_emergency_keywords", [])]
    urls_to_try = [feed_config["url"]]
    if feed_config.get("fallback_url"):
        urls_to_try.append(feed_config["fallback_url"])

    parsed = None
    for url in urls_to_try:
        logger.info(f"Fetching RSS: {feed_name} ({url})")
        raw = _fetch_raw(url) if REQUESTS_AVAILABLE else None
        if raw:
            parsed = feedparser.parse(raw)
        else:
            parsed = feedparser.parse(url)

        # Accept the result if we got entries, even if bozo
        if parsed and parsed.entries:
            logger.info(f"{feed_name}: {len(parsed.entries)} entries (bozo={parsed.bozo})")
            break
        else:
            bozo_exc = getattr(parsed, "bozo_exception", None) if parsed else None
            logger.warning(f"{feed_name} ({url}): no entries. bozo={getattr(parsed, 'bozo', False)}, exc={bozo_exc}")
            parsed = None

    if not parsed or not parsed.entries:
        logger.warning(f"{feed_name}: all URLs exhausted with no entries")
        return []

    alerts = []
    for entry in parsed.entries[:max_entries]:
        title = getattr(entry, "title", "").lower()
        summary = getattr(entry, "summary", getattr(entry, "description", "")).lower()
        combined = title + " " + summary

        # Check if India-relevant
        is_india = any(kw in combined for kw in india_kw)
        is_global_emergency = any(kw in combined for kw in global_kw)

        if india_only and not is_india and not is_global_emergency:
            continue
        if not is_india and not is_global_emergency:
            continue

        alert = parse_rss_entry_to_alert(entry, feed_name, source_tag)
        alerts.append(alert)

    logger.info(f"{feed_name}: {len(alerts)} relevant alerts parsed")
    return alerts


# ---------------------------------------------------------------------------
# Main Poller Class
# ---------------------------------------------------------------------------

class GovAlertPoller:
    """
    Polls all configured government/international alert feeds.
    Deduplicates by alert_id across sources.
    """

    def __init__(self, india_only: bool = True):
        self.india_only = india_only
        self._seen_ids: set = set()

    def poll_all_sources(self) -> List[Dict[str, Any]]:
        """
        Poll all configured RSS feeds and return a deduplicated list of alerts
        sorted by timestamp descending.
        """
        all_alerts: List[Dict[str, Any]] = []

        for feed_name, feed_config in FEEDS.items():
            try:
                feed_alerts = _fetch_feed(
                    feed_name=feed_name,
                    feed_config=feed_config,
                    india_only=self.india_only,
                )
                # Deduplicate
                for alert in feed_alerts:
                    aid = alert["alert_id"]
                    if aid not in self._seen_ids:
                        self._seen_ids.add(aid)
                        all_alerts.append(alert)
            except Exception as e:
                logger.error(f"Unexpected error polling {feed_name}: {e}")

        # Sort by fetch time descending (newest first)
        all_alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)

        logger.info(
            f"GovAlertPoller: {len(all_alerts)} total unique alerts from {len(FEEDS)} sources"
        )
        return all_alerts

    def reset(self):
        """Clear the deduplication cache so next poll fetches everything fresh."""
        self._seen_ids.clear()


# ---------------------------------------------------------------------------
# Module-level singleton + convenience function for use in main.py
# ---------------------------------------------------------------------------

_poller = GovAlertPoller(india_only=True)


def poll_government_alerts(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Convenience wrapper. If force_refresh=True, clears dedup cache first.
    """
    if force_refresh:
        _poller.reset()
    return _poller.poll_all_sources()
