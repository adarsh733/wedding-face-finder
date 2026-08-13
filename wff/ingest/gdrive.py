"""Google Drive public folder, read anonymously with our own API key.

WHY THIS FILE MATTERS (docs/ARCHITECTURE.md, landmine #2):

Reading a user's Drive needs the `drive.readonly` scope -- a RESTRICTED scope,
requiring a paid annual third-party security assessment. That only applies if
the photographer logs in with Google and grants us access to their account.

If instead they paste a link to a folder already set to "anyone with the link
can view", we read it as an anonymous member of the public using our own API
key. No login, no OAuth, no restricted scope, no audit.

>>> STATUS: HALF VERIFIED (13 Aug 2026). <<<
The SHARING half is confirmed on a real folder: Adarsh's link was fetched with
no cookies and no account and came back as the folder, not a sign-in wall and
not "request access". So a public folder really is readable by a stranger.

The API half is still unproven, because it needs a key and this machine had
none -- which is exactly why the first run died in under a second. Once a key
exists, `wff set-key <key>` then `wff verify-drive <link>` closes open item #1.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Any, BinaryIO, Iterator

import requests

from ..config import google_api_key
from .base import FileRef, IngestAdapter, ValidationResult, classify

DRIVE_API = "https://www.googleapis.com/drive/v3/files"

# Printed verbatim to whoever hit the wall. Nobody should have to find this in
# a source file, and "set WFF_GOOGLE_API_KEY" on its own tells them nothing.
NO_KEY_MESSAGE = (
    "Google needs a key before it will hand over a public folder, and this "
    "machine does not have one yet.\n"
    "  Get one (free, 2 minutes, no consent screen, no scopes):\n"
    "    1. console.cloud.google.com -> create a project (any name)\n"
    "    2. APIs & Services -> Library -> search 'Google Drive API' -> Enable\n"
    "    3. APIs & Services -> Credentials -> Create credentials -> API key\n"
    "    4. Paste it into the box on the console's 'Start a run' page.\n"
    "  This key reads public folders only. It is not a login and it cannot "
    "touch anyone's private files.\n"
    "  (From a terminal instead: python -m wff set-key <the-key>)"
)

# 403 means two completely different things here. Only these are "slow down".
_RETRYABLE_403 = {
    "rateLimitExceeded",
    "userRateLimitExceeded",
    "backendError",
    "internalError",
}

# A folder link comes in several shapes depending on how it was copied.
_FOLDER_PATTERNS = [
    re.compile(r"/folders/([A-Za-z0-9_-]{10,})"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]{10,})"),
    re.compile(r"/drive/u/\d+/folders/([A-Za-z0-9_-]{10,})"),
]

FOLDER_MIME = "application/vnd.google-apps.folder"
# Google Docs/Sheets/Slides have no bytes to download; they are never photos.
_GOOGLE_NATIVE_PREFIX = "application/vnd.google-apps."


def _google_error(response) -> tuple[str, str]:
    """(reason, message) out of Google's error envelope. Both may be empty."""
    try:
        error = response.json().get("error", {})
    except ValueError:
        return "", response.text[:300]
    errors = error.get("errors") or [{}]
    return str(errors[0].get("reason", "")), str(error.get("message", ""))


def explain_failure(response) -> str:
    """Google's error, said in words the photographer could act on."""
    reason, message = _google_error(response)
    status = response.status_code
    if status == 404:
        return (
            "Folder not found, or not shared publicly. Open it in Drive, press "
            "Share, and set it to 'Anyone with the link'."
        )
    if reason in ("accessNotConfigured",) or "has not been used in project" in message:
        return (
            "The key works, but the Google Drive API is switched off for its "
            "project. In console.cloud.google.com: APIs & Services -> Library "
            "-> Google Drive API -> Enable. It can take a minute to take effect."
            f" (Google said: {message[:200]})"
        )
    if status == 400 and ("API key not valid" in message or reason == "keyInvalid"):
        return (
            "That API key is not valid -- it may have been mistyped or deleted. "
            "Check for a missing character, or make a new one and paste it again."
        )
    if status == 403 and ("referer" in message.lower() or "restrict" in message.lower()):
        return (
            "That API key is restricted so it will not answer this program. In "
            "the key's settings, set Application restrictions to 'None' (or to "
            "IP addresses) -- a website/referrer restriction blocks us. "
            f"(Google said: {message[:200]})"
        )
    if status == 403:
        return (
            "Google refused the request. Usually this is the folder not being "
            "public, or the key being restricted. "
            f"(Google said: {message[:200] or reason})"
        )
    return f"Drive returned HTTP {status}: {message[:300] or response.text[:300]}"


def extract_folder_id(link: str) -> str | None:
    for pattern in _FOLDER_PATTERNS:
        match = pattern.search(link)
        if match:
            return match.group(1)
    # A bare id pasted on its own.
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", link.strip()):
        return link.strip()
    return None


class GoogleDriveFolderAdapter(IngestAdapter):
    source_type = "gdrive"

    def __init__(self, link: str, api_key: str | None = None) -> None:
        super().__init__(link)
        self.api_key = (api_key or google_api_key()).strip()
        self.folder_id = extract_folder_id(link)
        self._session = requests.Session()

    # -- polite, retrying HTTP --------------------------------------------
    def _get(self, url: str, params: dict[str, Any], stream: bool = False):
        """A 4,000-file job needs a polite, slow, retrying downloader.

        Google throttles hard. A tight loop gets us 403 rateLimitExceeded and
        then nothing.
        """
        delay = 1.0
        last_error = None
        for attempt in range(6):
            try:
                response = self._session.get(
                    url, params=params, stream=stream, timeout=(10, 120)
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            # A 403 is either "you are going too fast" or "no, and never" --
            # a wrong key, a restricted key, the Drive API not switched on.
            # Retrying the second kind burns a minute and then reports a
            # timeout, hiding the one sentence that would have fixed it.
            if response.status_code == 403:
                reason, _ = _google_error(response)
                if reason not in _RETRYABLE_403:
                    return response
            if response.status_code in (403, 429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                response.close()
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            return response
        raise RuntimeError(f"Drive request failed after retries: {last_error}")

    # -- interface ---------------------------------------------------------
    def validate(self) -> ValidationResult:
        if not self.api_key:
            return ValidationResult(False, NO_KEY_MESSAGE, {"needs_key": True})
        if not self.folder_id:
            return ValidationResult(
                False, f"Could not find a folder id in: {self.link!r}"
            )
        response = self._get(
            f"{DRIVE_API}/{self.folder_id}",
            {"key": self.api_key, "fields": "id,name,mimeType", "supportsAllDrives": "true"},
        )
        if response.status_code != 200:
            # The status is carried out so a caller can tell the two failures
            # apart: 404 means the KEY worked and the FOLDER is private, which
            # is no reason to throw away a good key.
            return ValidationResult(
                False, explain_failure(response), {"status": response.status_code}
            )
        payload = response.json()
        if payload.get("mimeType") != FOLDER_MIME:
            return ValidationResult(False, f"That link is not a folder: {payload.get('mimeType')}")
        return ValidationResult(
            True, f"Public folder readable: {payload.get('name')!r}", {"name": payload.get("name")}
        )

    def _list_children(self, folder_id: str) -> Iterator[dict]:
        page_token = None
        while True:
            params = {
                "key": self.api_key,
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken,files(id,name,mimeType,size)",
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            response = self._get(DRIVE_API, params)
            if response.status_code != 200:
                # Never raise_for_status here: requests puts the full URL in the
                # message, and the URL carries the API key. That key would then
                # be in the run log, in runs.jsonl, and on screen.
                raise RuntimeError(explain_failure(response))
            payload = response.json()
            yield from payload.get("files", [])
            page_token = payload.get("nextPageToken")
            if not page_token:
                return

    def list_files(self) -> Iterator[FileRef]:
        if not self.folder_id:
            raise ValueError(f"No folder id in link: {self.link!r}")
        # Iterative walk, so a deeply nested Haldi/Day1/Morning/ cannot blow the
        # Python stack.
        stack: list[tuple[str, str]] = [(self.folder_id, "")]
        seen: set[str] = set()
        while stack:
            folder_id, prefix = stack.pop()
            if folder_id in seen:
                continue  # Drive allows a file in two parents; do not loop.
            seen.add(folder_id)
            for item in self._list_children(folder_id):
                name = item["name"]
                path = f"{prefix}{name}"
                if item.get("mimeType") == FOLDER_MIME:
                    stack.append((item["id"], f"{path}/"))
                    continue
                if str(item.get("mimeType", "")).startswith(_GOOGLE_NATIVE_PREFIX):
                    continue
                size = item.get("size")
                yield FileRef(
                    source_id=item["id"],
                    name=name,
                    path=path,
                    kind=classify(name),
                    size=int(size) if size is not None else None,
                    mime=item.get("mimeType"),
                )

    @contextmanager
    def open(self, ref: FileRef) -> Iterator[BinaryIO]:
        response = self._get(
            f"{DRIVE_API}/{ref.source_id}",
            {"key": self.api_key, "alt": "media", "supportsAllDrives": "true"},
            stream=True,
        )
        if response.status_code != 200:
            response.close()
            raise RuntimeError(f"{ref.path}: {explain_failure(response)}")
        try:
            response.raw.decode_content = True
            yield response.raw
        finally:
            response.close()

    def source_folder_id(self) -> str:
        return self.folder_id or self.link
