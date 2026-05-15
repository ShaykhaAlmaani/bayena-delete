"""
Tests for the dataset selector / interpret_request logic.
"""

import pytest
from app.core.guardrails import check_analyst_question, sanitize_user_prompt, validate_prompt_length


class TestSanitizePrompt:
    def test_removes_null_bytes(self):
        result = sanitize_user_prompt("hello\x00world")
        assert "\x00" not in result

    def test_trims_whitespace(self):
        result = sanitize_user_prompt("  air quality  ")
        assert result == "air quality"

    def test_caps_at_2000_chars(self):
        long = "a" * 3000
        result = sanitize_user_prompt(long)
        assert len(result) == 2000

    def test_empty_string(self):
        assert sanitize_user_prompt("") == ""


class TestValidatePromptLength:
    def test_too_short(self):
        valid, msg = validate_prompt_length("hi")
        assert not valid
        assert msg

    def test_valid_prompt(self):
        valid, msg = validate_prompt_length("Show me air quality trends in Riyadh")
        assert valid
        assert msg == ""

    def test_too_long(self):
        valid, msg = validate_prompt_length("a" * 2001)
        assert not valid


class TestCheckAnalystQuestion:
    def test_valid_question(self):
        allowed, _ = check_analyst_question("Which region has the most violations?")
        assert allowed

    def test_blocks_political(self):
        allowed, msg = check_analyst_question("Tell me about politics in Saudi Arabia")
        assert not allowed
        assert "dashboard" in msg.lower()

    def test_blocks_code_request(self):
        allowed, msg = check_analyst_question("Write me a Python script using import os")
        assert not allowed

    def test_blocks_edit_request(self):
        allowed, msg = check_analyst_question("Change the second chart to a bar chart")
        assert not allowed
        assert "Edit Dashboard" in msg

    def test_empty_question(self):
        allowed, msg = check_analyst_question("")
        assert not allowed

    def test_too_long_question(self):
        allowed, msg = check_analyst_question("q" * 1001)
        assert not allowed
