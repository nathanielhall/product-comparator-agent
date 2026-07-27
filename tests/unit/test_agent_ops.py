# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging

import pytest

from app.agent import (
    JsonFormatter,
    SearchInput,
    comparison_agent,
    google_search,
    redact_pii,
    research_agent,
    save_session_memory_async,
    search_helper_agent,
)


def test_strategic_model_routing() -> None:
    """Verify that agents use strategic model routing configurations."""
    assert comparison_agent.model.model == "gemini-2.5-pro"
    assert search_helper_agent.model.model == "gemini-3.6-flash"
    assert research_agent.model.model == "gemini-3.6-flash"


def test_pii_redaction() -> None:
    """Verify that redact_pii scrubs SSNs, credit cards, and email patterns."""
    raw_query = "Compare iPhone 15 for john.doe@example.com with SSN 123-45-6789 and card 4532-1234-5678-9012"
    sanitized = redact_pii(raw_query)

    assert "[REDACTED_EMAIL]" in sanitized
    assert "john.doe@example.com" not in sanitized
    assert "[REDACTED_SSN]" in sanitized
    assert "123-45-6789" not in sanitized
    assert "[REDACTED_CREDIT_CARD]" in sanitized
    assert "4532-1234-5678-9012" not in sanitized


def test_google_search_tool_valid_and_error_handling() -> None:
    """Verify google_search tool schema, successful execution, and natural language error recovery."""
    inp = SearchInput(query="Pixel 8 Pro")
    res = google_search(inp)
    assert "Search results for 'Pixel 8 Pro'" in res

    # Empty query guided recovery instruction
    empty_inp = SearchInput(query="   ")
    err_res = google_search(empty_inp)
    assert "Search failed:" in err_res
    assert "Ask the user for clarification" in err_res


@pytest.mark.asyncio
async def test_save_session_memory_async() -> None:
    """Verify background async memory persistence function completes without error."""
    await save_session_memory_async("Sample comparison summary")


def test_json_formatter() -> None:
    """Verify JsonFormatter outputs structured JSON logs."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    formatted_output = formatter.format(record)
    data = json.loads(formatted_output)

    assert data["logger"] == "test_logger"
    assert data["level"] == "INFO"
    assert data["message"] == "Test log message"
    assert "timestamp" in data
