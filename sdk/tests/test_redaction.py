"""Redaction: coverage, false-positive control, extensibility, fail-closed."""

from __future__ import annotations

import re

import pytest

from agentlens.redaction import (
    DEFAULT_REDACTOR,
    KNOWN_LIMITATIONS,
    RedactionRule,
    Redactor,
    build_redactor,
    _luhn_ok,
)


@pytest.mark.parametrize(
    "text,rule",
    [
        ("contact me at alice.smith@example.com please", "email"),
        ("ssn 123-45-6789 on file", "us_ssn"),
        ("card 4111 1111 1111 1111 charged", "credit_card"),
        ("server at 192.168.1.100 responded", "ipv4"),
        ("host 2001:0db8:85a3:0000:0000:8a2e:0370:7334", "ipv6"),
        ("key AKIAIOSFODNN7EXAMPLE leaked", "aws_access_key_id"),
        ("Authorization: Bearer abcdefghijklmnop123456", "authorization_header"),
        ("call +1 555-123-4567 now", "phone"),
    ],
)
def test_default_rules_detect_common_pii(text: str, rule: str) -> None:
    outcome = DEFAULT_REDACTOR.redact(text)
    assert rule in outcome.rules_fired, outcome
    assert "[REDACTED:" in outcome.text


def test_jwt_is_redacted_whole() -> None:
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    outcome = DEFAULT_REDACTOR.redact(f"token={jwt}")
    assert "eyJ" not in outcome.text
    assert "jwt" in outcome.rules_fired


def test_private_key_block_is_redacted_including_body() -> None:
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAvBkYdF3sQ0\nsecretmaterialhere\n"
        "-----END RSA PRIVATE KEY-----"
    )
    outcome = DEFAULT_REDACTOR.redact(f"here it is:\n{key}\nend")
    assert "secretmaterialhere" not in outcome.text
    assert "private_key_block" in outcome.rules_fired


def test_anthropic_key_is_redacted() -> None:
    outcome = DEFAULT_REDACTOR.redact("sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA")
    assert "anthropic_api_key" in outcome.rules_fired
    assert "sk-ant-api03" not in outcome.text


# -- false positives --------------------------------------------------------


def test_luhn_gate_rejects_non_card_digit_runs() -> None:
    """A long digit run that fails Luhn is not a card and must survive."""
    # 16 digits, deliberately not Luhn-valid.
    assert not _luhn_ok("1234567812345678")
    outcome = DEFAULT_REDACTOR.redact("order reference 1234567812345678")
    assert "credit_card" not in outcome.rules_fired
    assert "1234567812345678" in outcome.text


def test_luhn_gate_accepts_a_valid_card() -> None:
    assert _luhn_ok("4111111111111111")
    outcome = DEFAULT_REDACTOR.redact("4111111111111111")
    assert "credit_card" in outcome.rules_fired


def test_serialised_floats_are_never_matched_as_credit_cards() -> None:
    """Regression: the card rule was destroying retrieval scores.

    Digits after a decimal point form a 13-19 digit run, and ~1 in 10 passes
    Luhn by chance. Measured on the real Clinical RAG trace this corrupted
    retrieval scores; a sweep of 20,000 random floats reproduced it at 9.49%.
    Scores, probabilities and latencies are the data this platform measures, so
    a false positive here is worse than a miss.
    """
    import random

    random.seed(7)
    false_positives = 0
    for _ in range(20_000):
        payload = f'{{"score": {random.random()!r}}}'
        if "credit_card" in DEFAULT_REDACTOR.redact(payload).rules_fired:
            false_positives += 1
    assert false_positives == 0, (
        f"{false_positives}/20000 floats matched as credit cards"
    )


@pytest.mark.parametrize(
    "text",
    [
        "4111111111111111",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "pay with 4111-1111-1111-1111.",
    ],
)
def test_real_cards_still_detected_after_the_float_fix(text: str) -> None:
    """The lookbehind must not have cost us true positives."""
    outcome = DEFAULT_REDACTOR.redact(text)
    assert "credit_card" in outcome.rules_fired
    assert "4111" not in outcome.text


def test_bare_digit_run_is_not_treated_as_a_phone_number() -> None:
    outcome = DEFAULT_REDACTOR.redact("the id is 5551234567")
    assert "phone" not in outcome.rules_fired


def test_ordinary_text_is_untouched_and_reports_no_rules() -> None:
    text = "Summarise the quarterly revenue trend for the northeast region."
    outcome = DEFAULT_REDACTOR.redact(text)
    assert outcome.text == text
    assert outcome.rules_fired == ()
    assert not outcome.changed


# -- extensibility ----------------------------------------------------------


def test_custom_rule_can_be_added() -> None:
    rule = RedactionRule(
        name="employee_id",
        pattern=re.compile(r"\bEMP-\d{6}\b"),
        replacement="[REDACTED:employee_id]",
    )
    redactor = DEFAULT_REDACTOR.with_rules(rule)
    outcome = redactor.redact("ticket raised by EMP-004512 today")
    assert "employee_id" in outcome.rules_fired
    assert "EMP-004512" not in outcome.text


def test_custom_rule_can_take_priority_over_defaults() -> None:
    """prepend=True lets a domain rule claim text a default would consume."""
    rule = RedactionRule(
        name="internal_mailbox",
        pattern=re.compile(r"\b\w+@internal\.example\.com\b"),
        replacement="[REDACTED:internal_mailbox]",
    )
    redactor = DEFAULT_REDACTOR.with_rules(rule, prepend=True)
    outcome = redactor.redact("ping ops@internal.example.com")
    assert "internal_mailbox" in outcome.rules_fired
    assert "email" not in outcome.rules_fired


def test_rules_can_be_disabled() -> None:
    redactor = DEFAULT_REDACTOR.without_rules("ipv4")
    outcome = redactor.redact("host 10.0.0.5")
    assert "ipv4" not in outcome.rules_fired
    assert "10.0.0.5" in outcome.text


def test_arbitrary_transform_is_supported() -> None:
    redactor = DEFAULT_REDACTOR.with_transform(lambda t: t.replace("classified", "***"))
    outcome = redactor.redact("this is classified material")
    assert "***" in outcome.text
    assert "custom_transform" in outcome.rules_fired


def test_build_redactor_composes_options() -> None:
    redactor = build_redactor(
        extra_rules=[
            RedactionRule("badge", re.compile(r"BADGE\d+"), "[REDACTED:badge]")
        ],
        disable=["ipv4"],
        max_length=64,
    )
    outcome = redactor.redact("BADGE123 at 10.0.0.1 " + "x" * 200)
    assert "badge" in outcome.rules_fired
    assert "ipv4" not in outcome.rules_fired
    assert "truncated" in outcome.rules_fired
    assert len(outcome.text) < 200


def test_default_redactor_is_immutable_across_callers() -> None:
    """Customisation must not leak into the shared default."""
    before = len(DEFAULT_REDACTOR.rules)
    DEFAULT_REDACTOR.with_rules(
        RedactionRule("x", re.compile("x"), "y")
    ).without_rules("email")
    assert len(DEFAULT_REDACTOR.rules) == before


def test_truncation_reports_original_length() -> None:
    redactor = Redactor(max_length=20)
    outcome = redactor.redact("y" * 100)
    assert "truncated 100 chars" in outcome.text


def test_sequence_values_are_redacted_elementwise() -> None:
    value, rules = DEFAULT_REDACTOR.redact_value(["a@b.com", "plain", "c@d.com"])
    assert "email" in rules
    assert all("@" not in v for v in value if isinstance(v, str))


def test_non_text_values_pass_through_unchanged() -> None:
    for value in (42, 3.5, True, None):
        out, rules = DEFAULT_REDACTOR.redact_value(value)
        assert out is value
        assert rules == ()


def test_limitations_are_documented() -> None:
    """The gaps are stated, not omitted -- an operator must be able to judge fit."""
    assert len(KNOWN_LIMITATIONS) >= 3
    assert any("name" in limitation.lower() for limitation in KNOWN_LIMITATIONS)


def test_multiple_rules_can_fire_on_one_value() -> None:
    outcome = DEFAULT_REDACTOR.redact("mail a@b.com from 10.1.2.3 ssn 111-22-3333")
    assert {"email", "ipv4", "us_ssn"} <= set(outcome.rules_fired)
