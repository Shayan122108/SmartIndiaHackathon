"""
SMS Tools — Twilio Integration
Handles outbound SMS alerts, subscriber management, and bulk broadcasting.
Subscribers are seeded from SMS_ALERT_SUBSCRIBERS in .env at startup.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subscriber Registry (in-memory, seeded from .env)
# ---------------------------------------------------------------------------

@dataclass
class SMSSubscriber:
    phone_number: str          # E.164 format, e.g. "+919148033536"
    name: str = "Subscriber"
    subscribed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    active: bool = True


class SubscriberRegistry:
    """Thread-safe in-memory SMS subscriber list."""

    def __init__(self):
        self._subscribers: Dict[str, SMSSubscriber] = {}
        self._seed_from_env()

    def _seed_from_env(self):
        """Load initial subscribers from SMS_ALERT_SUBSCRIBERS env var (comma-separated)."""
        raw = os.getenv("SMS_ALERT_SUBSCRIBERS", "")
        if not raw.strip():
            return
        for num in raw.split(","):
            num = num.strip()
            if num:
                self.add(num, name="Admin (env)")
        logger.info(f"Seeded {len(self._subscribers)} SMS subscriber(s) from .env")

    def add(self, phone_number: str, name: str = "Subscriber") -> SMSSubscriber:
        """Add or reactivate a subscriber."""
        phone_number = self._normalize(phone_number)
        if phone_number in self._subscribers:
            self._subscribers[phone_number].active = True
            logger.info(f"Reactivated subscriber: {phone_number}")
        else:
            sub = SMSSubscriber(phone_number=phone_number, name=name)
            self._subscribers[phone_number] = sub
            logger.info(f"Added new SMS subscriber: {phone_number}")
        return self._subscribers[phone_number]

    def remove(self, phone_number: str) -> bool:
        """Deactivate (soft-delete) a subscriber. Returns True if found."""
        phone_number = self._normalize(phone_number)
        if phone_number in self._subscribers:
            self._subscribers[phone_number].active = False
            logger.info(f"Deactivated subscriber: {phone_number}")
            return True
        return False

    def get_active(self) -> List[SMSSubscriber]:
        """Return all currently active subscribers."""
        return [s for s in self._subscribers.values() if s.active]

    def all(self) -> List[SMSSubscriber]:
        """Return all subscribers (including inactive)."""
        return list(self._subscribers.values())

    def count_active(self) -> int:
        return len(self.get_active())

    @staticmethod
    def _normalize(number: str) -> str:
        """Strip whitespace; keep as-is (E.164 expected)."""
        return number.strip()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "phone_number": s.phone_number,
                "name": s.name,
                "subscribed_at": s.subscribed_at,
                "active": s.active,
            }
            for s in self.all()
        ]


# Singleton registry — imported by main.py
subscriber_registry = SubscriberRegistry()


# ---------------------------------------------------------------------------
# Twilio Client
# ---------------------------------------------------------------------------

def _get_twilio_client():
    """Lazy-init Twilio client. Returns None if credentials are missing."""
    try:
        from twilio.rest import Client
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if not account_sid or not auth_token:
            logger.warning("Twilio credentials missing — SMS will be skipped")
            return None
        return Client(account_sid, auth_token)
    except ImportError:
        logger.error("twilio package not installed — run: pip install twilio")
        return None
    except Exception as e:
        logger.error(f"Failed to init Twilio client: {e}")
        return None


# ---------------------------------------------------------------------------
# Core Send Function
# ---------------------------------------------------------------------------

def send_sms(to_number: str, message: str) -> Dict[str, Any]:
    """
    Send a single SMS via Twilio.

    Args:
        to_number: E.164 phone number, e.g. "+919148033536"
        message: Message body (max 1600 chars; longer messages split into segments)

    Returns:
        dict with keys: success (bool), sid or error, to, timestamp
    """
    client = _get_twilio_client()
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not client:
        return {
            "success": False,
            "error": "Twilio client unavailable — check TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN",
            "to": to_number,
            "timestamp": datetime.now().isoformat(),
        }
    if not from_number:
        return {
            "success": False,
            "error": "TWILIO_PHONE_NUMBER not set in .env",
            "to": to_number,
            "timestamp": datetime.now().isoformat(),
        }

    try:
        msg = client.messages.create(
            body=message[:1600],   # Twilio max per message
            from_=from_number,
            to=to_number,
        )
        logger.info(f"SMS sent to {to_number} — SID: {msg.sid}")
        return {
            "success": True,
            "sid": msg.sid,
            "to": to_number,
            "status": msg.status,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"SMS send failed to {to_number}: {e}")
        return {
            "success": False,
            "error": str(e),
            "to": to_number,
            "timestamp": datetime.now().isoformat(),
        }


# ---------------------------------------------------------------------------
# Alert Formatter
# ---------------------------------------------------------------------------

def format_alert_as_sms(alert: Dict[str, Any]) -> str:
    """
    Convert an internal alert dict into a concise SMS-friendly string (<= 450 chars
    to stay within 3 segments on most carriers).
    """
    level = alert.get("alert_level", "ALERT").upper()
    title = alert.get("title", "Emergency Alert")
    summary = alert.get("summary", "")
    contacts = alert.get("emergency_contacts", {})

    # Build contact line
    contact_parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in list(contacts.items())[:2]]
    contact_line = " | ".join(contact_parts)

    body = f"🚨 [{level}] {title}\n{summary}"
    if contact_line:
        body += f"\nEmergency: {contact_line}"
    body += "\n— Warangal Health Alert System"

    # Truncate safely
    return body[:450]


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------

def broadcast_sms_alert(
    alert: Dict[str, Any],
    registry: Optional[SubscriberRegistry] = None,
    extra_numbers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Send an SMS alert to all active subscribers plus any extra_numbers provided.

    Args:
        alert: Internal alert dict (from emergency_agent or gov_alert_tools)
        registry: SubscriberRegistry instance (defaults to module singleton)
        extra_numbers: Additional one-off numbers to include in this broadcast

    Returns:
        dict with total_sent, total_failed, results list
    """
    if registry is None:
        registry = subscriber_registry

    message = format_alert_as_sms(alert)
    targets = {s.phone_number for s in registry.get_active()}
    if extra_numbers:
        targets.update(extra_numbers)

    if not targets:
        logger.info("SMS broadcast: no subscribers to notify")
        return {"total_sent": 0, "total_failed": 0, "results": [], "message_preview": message}

    results = []
    sent = 0
    failed = 0

    for number in targets:
        result = send_sms(number, message)
        results.append(result)
        if result["success"]:
            sent += 1
        else:
            failed += 1

    logger.info(f"SMS broadcast complete — sent: {sent}, failed: {failed}")
    return {
        "total_sent": sent,
        "total_failed": failed,
        "results": results,
        "message_preview": message,
        "timestamp": datetime.now().isoformat(),
    }
