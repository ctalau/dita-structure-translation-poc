"""Context-aware French typography oracle."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dita_reward import count_violations, typography_oracle  # noqa: E402
from fr_typography_fixtures import build_cases  # noqa: E402
from fr_typography_postprocess import postprocess_french_typography  # noqa: E402

REQUIRED = {
    "positives": 40,
    "negatives": 40,
    "inline_tails": 8,
    "entities": 6,
    "clusters": 8,
    "omitted_marks": 8,
    "deleted_prose": 8,
    "identifier_mutation": 4,
    "wrapper_insertion": 6,
    "invalid_xml": 6,
    "sibling_swaps": 4,
    "protected_literals": 8,
    "adversarial": 4,
    "ambiguous": 4,
    "guillemets": 4,
    "nfkc": 2,
}


def _score_all():
    cases = build_cases()
    scored = {}
    for case in cases:
        scored[case["id"]] = typography_oracle(case["source"], case["hyp"])
    return cases, scored


CASES, SCORED = _score_all()


def test_fixture_count_and_categories():
    assert len(CASES) >= 250
    assert len({case["id"] for case in CASES}) == len(CASES)
    counts = Counter(case["category"] for case in CASES)
    for name, minimum in REQUIRED.items():
        assert counts[name] >= minimum, (name, counts[name])


def test_expectations_and_no_signoff():
    for case in CASES:
        result = SCORED[case["id"]]
        assert result["evaluator_version"] == "fr-typography-oracle-1"
        assert result["human_french_review_signed_off"] is False
        assert result["reviewer"] is None
        assert result["reward_uses_nbsp_count"] is False
        if "expect_reward" in case:
            assert result["reward"] == case["expect_reward"], (case["id"], result["reward"])
        if "expect_gate" in case:
            assert result["gate_passed"] is case["expect_gate"], (case["id"], result["gate_failures"])
        if case.get("expect_ambiguous"):
            assert result["needs_human_french_review"] is True
            assert result["ambiguous_spans"]
            assert all(span["reviewer_signed_off"] is False for span in result["ambiguous_spans"])
        else:
            assert result["needs_human_french_review"] is False or result["ambiguous_spans"] == [] or case.get("expect_ambiguous")
    for case in CASES:
        if "reward_lt_id" in case:
            assert SCORED[case["id"]]["reward"] < SCORED[case["reward_lt_id"]]["reward"], case["id"]
        if "reward_le_id" in case:
            assert SCORED[case["id"]]["reward"] <= SCORED[case["reward_le_id"]]["reward"], case["id"]


def test_invalid_below_every_valid_floor():
    valid_rewards = [SCORED[case["id"]]["reward"] for case in CASES if SCORED[case["id"]]["gate_passed"]]
    invalid_rewards = [SCORED[case["id"]]["reward"] for case in CASES if not SCORED[case["id"]]["gate_passed"]]
    assert valid_rewards
    assert invalid_rewards
    assert min(valid_rewards) >= 0.0
    assert max(invalid_rewards) < min(valid_rewards)


def test_attacks_do_not_improve_reward():
    assert SCORED["wrap-codeph"]["reward"] < SCORED["neg-:-space-0"]["reward"]
    assert SCORED["adv-codeph-hides-colon"]["reward"] < SCORED["neg-:-space-0"]["reward"]
    assert SCORED["delete-sentence"]["reward"] <= SCORED["neg-:-space-0"]["reward"]
    assert SCORED["protect-xs-spaced"]["reward"] < SCORED["protect-xs-glued"]["reward"]
    assert SCORED["adv-space-xs"]["reward"] < SCORED["adv-xs-glued-with-mark"]["reward"]
    assert SCORED["swap-li"]["reward"] < SCORED["swap-li-ordered"]["reward"]
    assert SCORED["mutate-xs"]["reward"] < SCORED["protect-xs-glued"]["reward"]


def test_old_counter_differs_on_namespace_and_inline_boundary():
    glued = next(case for case in CASES if case["id"] == "protect-xs-glued")
    old = count_violations(glued["source"], glued["hyp"])
    assert old["punct"] > 0
    assert SCORED["protect-xs-glued"]["reward"] == 1.0
    inline = next(case for case in CASES if case["id"] == "inline-end-nbsp")
    old_inline = count_violations(inline["source"], inline["hyp"])
    assert old_inline["punct"] > 0
    assert SCORED["inline-end-nbsp"]["reward"] == 1.0


def test_nbsp_count_is_not_a_reward_term():
    extra = SCORED["adv-extra-nbsp-same-reward"]
    base = SCORED["pos-:-nbsp-0"]
    assert extra["nbsp_count"] > base["nbsp_count"]
    assert extra["reward"] == base["reward"]
    assert SCORED["cluster-extra-nbsp"]["nbsp_count"] > SCORED["cluster-ok"]["nbsp_count"]
    assert SCORED["cluster-extra-nbsp"]["reward"] == SCORED["cluster-ok"]["reward"]


def test_postprocess_baseline_on_negatives_and_literals():
    improved = 0
    regressed = 0
    for case in CASES:
        if case["category"] not in ("negatives", "protected_literals", "guillemets"):
            continue
        if not SCORED[case["id"]]["gate_passed"]:
            continue
        edited = postprocess_french_typography(case["hyp"])
        assert "<codeph>" not in edited or "<codeph>" in case["hyp"]
        assert "xs\u00A0:" not in edited
        assert "oxy\u00A0:" not in edited
        after = typography_oracle(case["source"], edited)
        if after["reward"] > SCORED[case["id"]]["reward"]:
            improved += 1
        elif after["reward"] < SCORED[case["id"]]["reward"]:
            regressed += 1
    assert regressed == 0
    assert improved > 0
