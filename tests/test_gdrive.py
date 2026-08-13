"""The Drive path, tested where it actually broke.

A real public folder link was pasted into the console on 13 Aug and the run
died in 0.0s: no API key. Everything here guards the two lessons from that --
the key must survive a closed terminal, and a refusal from Google must arrive
as a sentence someone can act on rather than a timeout.
"""
from __future__ import annotations

import json
import os

import pytest

from wff import config
from wff.ingest import adapter_for
from wff.ingest.gdrive import (
    GoogleDriveFolderAdapter,
    explain_failure,
    extract_folder_id,
)


class FakeResponse:
    """Just enough of requests.Response for the error paths."""

    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def close(self):
        pass


def google_error(reason: str, message: str, status: int = 403) -> FakeResponse:
    return FakeResponse(
        status,
        {"error": {"code": status, "message": message, "errors": [{"reason": reason}]}},
    )


# -- the link the photographer pastes ---------------------------------------

def test_folder_id_out_of_the_link_adarsh_actually_pasted():
    link = "https://drive.google.com/drive/folders/1oBr6vhYfIQoHm8ITnAJvfFCZi-rgu5sQ"
    assert extract_folder_id(link) == "1oBr6vhYfIQoHm8ITnAJvfFCZi-rgu5sQ"
    assert adapter_for(link).source_type == "gdrive"


def test_folder_id_survives_the_shapes_a_link_gets_copied_in():
    fid = "1oBr6vhYfIQoHm8ITnAJvfFCZi-rgu5sQ"
    for link in (
        f"https://drive.google.com/drive/u/0/folders/{fid}",
        f"https://drive.google.com/drive/folders/{fid}?usp=sharing",
        f"https://drive.google.com/open?id={fid}",
        fid,
    ):
        assert extract_folder_id(link) == fid, link


# -- the key -----------------------------------------------------------------

def test_no_key_says_how_to_get_one(monkeypatch):
    """The old message named an environment variable and stopped there."""
    monkeypatch.delenv("WFF_GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config, "_settings_loaded", True)  # ignore this machine's file
    check = GoogleDriveFolderAdapter("https://drive.google.com/drive/folders/abcdefghij").validate()
    assert not check.ok
    assert "set-key" in check.message
    assert "console.cloud.google.com" in check.message


def test_key_saved_once_is_found_by_a_later_process(monkeypatch, tmp_path):
    """The whole point: a new terminal, or the console started from a shortcut."""
    settings = tmp_path / ".wff" / "settings.env"
    monkeypatch.setattr(config, "SETTINGS_FILE", str(settings))
    monkeypatch.setattr(config, "_PROJECT_ENV", str(tmp_path / "no-such.env"))
    monkeypatch.delenv("WFF_GOOGLE_API_KEY", raising=False)

    config.save_google_api_key("AIzaTESTKEY123")
    # Simulate a fresh process: nothing in the environment, nothing loaded.
    monkeypatch.delenv("WFF_GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config, "_settings_loaded", False)
    assert config.google_api_key() == "AIzaTESTKEY123"


def test_saving_a_key_keeps_the_other_settings(monkeypatch, tmp_path):
    settings = tmp_path / "settings.env"
    settings.write_text("WFF_DET_SIZE=1280\nWFF_GOOGLE_API_KEY=old\n", encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_FILE", str(settings))
    config.save_google_api_key("new")
    body = settings.read_text(encoding="utf-8")
    assert "WFF_DET_SIZE=1280" in body
    assert "WFF_GOOGLE_API_KEY=new" in body
    assert "old" not in body


def test_a_real_environment_variable_beats_the_file(monkeypatch, tmp_path):
    """A one-off override for testing must not be silently replaced."""
    settings = tmp_path / "settings.env"
    settings.write_text("WFF_GOOGLE_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("WFF_GOOGLE_API_KEY", "from-env")
    config._read_env_file(str(settings))
    assert os.environ["WFF_GOOGLE_API_KEY"] == "from-env"


# -- what Google says, in words ---------------------------------------------

def test_drive_api_switched_off_is_explained_not_echoed():
    response = google_error(
        "accessNotConfigured",
        "Google Drive API has not been used in project 12345 before or it is disabled.",
    )
    message = explain_failure(response)
    assert "Enable" in message and "Library" in message


def test_a_private_folder_is_told_to_be_shared():
    message = explain_failure(FakeResponse(404, {"error": {"message": "File not found"}}))
    assert "Anyone with the link" in message


def test_a_restricted_key_names_the_setting_to_change():
    response = google_error(
        "forbidden", "Requests from referer <empty> are blocked.", status=403
    )
    assert "restrict" in explain_failure(response).lower()


def test_an_invalid_key_says_so():
    response = FakeResponse(
        400, {"error": {"message": "API key not valid. Please pass a valid API key."}}
    )
    assert "not valid" in explain_failure(response)


# -- the retry loop ----------------------------------------------------------

def _adapter_returning(monkeypatch, response, calls):
    adapter = GoogleDriveFolderAdapter(
        "https://drive.google.com/drive/folders/1oBr6vhYfIQoHm8ITnAJvfFCZi-rgu5sQ",
        api_key="AIzaTEST",
    )

    def fake_get(url, params=None, stream=False, timeout=None):
        calls.append(url)
        return response

    monkeypatch.setattr(adapter._session, "get", fake_get)
    monkeypatch.setattr("wff.ingest.gdrive.time.sleep", lambda _s: calls.append("slept"))
    return adapter


def test_a_permanent_403_is_not_retried_for_a_minute(monkeypatch):
    """It used to retry 6 times over 63s and then report a timeout, burying
    the one sentence that would have fixed it."""
    calls: list[str] = []
    adapter = _adapter_returning(
        monkeypatch,
        google_error("accessNotConfigured", "Google Drive API has not been used"),
        calls,
    )
    check = adapter.validate()
    assert not check.ok
    assert "slept" not in calls
    assert len(calls) == 1
    assert "Enable" in check.message


def test_a_rate_limit_403_is_still_retried(monkeypatch):
    calls: list[str] = []
    adapter = _adapter_returning(
        monkeypatch, google_error("rateLimitExceeded", "Rate Limit Exceeded"), calls
    )
    with pytest.raises(RuntimeError):
        adapter.validate()
    assert calls.count("slept") >= 5


def test_the_api_key_never_reaches_an_error_message(monkeypatch):
    """requests' own raise_for_status prints the URL -- and the key is in it,
    so it would land in the run log and on screen."""
    calls: list[str] = []
    adapter = _adapter_returning(
        monkeypatch, google_error("forbidden", "The caller does not have permission"), calls
    )
    with pytest.raises(RuntimeError) as caught:
        list(adapter.list_files())
    assert "AIzaTEST" not in str(caught.value)


def test_a_folder_that_is_not_a_folder_is_rejected(monkeypatch):
    calls: list[str] = []
    adapter = _adapter_returning(
        monkeypatch,
        FakeResponse(200, {"id": "x", "name": "photo.jpg", "mimeType": "image/jpeg"}),
        calls,
    )
    check = adapter.validate()
    assert not check.ok
    assert "not a folder" in check.message
