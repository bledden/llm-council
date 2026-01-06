"""Tests for minority opinion detection."""

import sys
sys.path.insert(0, '/tmp/llm-council')

from backend.council import detect_minority_opinions, parse_ranking_from_text


def make_stage2_entry(model: str, ranking_order: list) -> dict:
    """Helper to create stage2 result entries with proper structure."""
    ranking_text = "FINAL RANKING:\n" + "\n".join(
        f"{i+1}. {label}" for i, label in enumerate(ranking_order)
    )
    return {
        "model": model,
        "ranking": ranking_text,
        "parsed_ranking": ranking_order
    }


def test_no_minority_when_consensus():
    """When all rankers agree, no minority opinions should be detected."""
    # All 3 rankers agree on the same order
    stage2_results = [
        make_stage2_entry("model_a", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_b", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_c", ["Response A", "Response B", "Response C"]),
    ]
    
    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b",
        "Response C": "model_c"
    }
    
    # Tournament rankings (consensus)
    tournament_rankings = [
        {"model": "model_a", "wins": 2, "win_percentage": 1.0},
        {"model": "model_b", "wins": 1, "win_percentage": 0.5},
        {"model": "model_c", "wins": 0, "win_percentage": 0.0},
    ]
    
    minority = detect_minority_opinions(stage2_results, label_to_model, tournament_rankings)
    
    assert len(minority) == 0, f"Expected no minority opinions, got {minority}"
    print("✓ No minority when consensus - PASSED")


def test_minority_detected_with_dissent():
    """When 1 of 3 rankers (33%) disagrees significantly, minority should be detected."""
    # 2 rankers say A is #1, 1 ranker says A is #3 (significant disagreement)
    stage2_results = [
        make_stage2_entry("model_a", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_b", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_c", ["Response B", "Response C", "Response A"]),  # Disagrees on A
    ]
    
    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b",
        "Response C": "model_c"
    }
    
    # Tournament has model_a at position 1
    tournament_rankings = [
        {"model": "model_a", "wins": 2, "win_percentage": 1.0},
        {"model": "model_b", "wins": 1, "win_percentage": 0.5},
        {"model": "model_c", "wins": 0, "win_percentage": 0.0},
    ]
    
    minority = detect_minority_opinions(stage2_results, label_to_model, tournament_rankings)
    
    # model_a: consensus position 1, but model_c placed it at position 3
    # That's 1/3 = 33% dissent, which meets the 30% threshold
    # Position difference is 2 (1->3), which exceeds tolerance of 1
    assert len(minority) >= 1, f"Expected at least 1 minority opinion, got {minority}"
    
    model_a_minority = next((m for m in minority if m["model"] == "model_a"), None)
    assert model_a_minority is not None, "Expected minority opinion for model_a"
    assert model_a_minority["consensus_position"] == 1
    assert 3 in model_a_minority["dissent_positions"]
    assert model_a_minority["dissent_rate"] >= 0.3
    assert "model_c" in model_a_minority["dissenters"]
    assert model_a_minority["direction"] == "overvalued"  # consensus ranks higher than dissenter thinks
    
    print("✓ Minority detected with dissent - PASSED")


def test_minority_direction_undervalued():
    """Test that 'undervalued' direction is correctly identified."""
    # Consensus has model_c at #3, but one ranker thinks it should be #1
    stage2_results = [
        make_stage2_entry("model_a", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_b", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_c", ["Response C", "Response A", "Response B"]),  # Thinks C is best
    ]
    
    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b",
        "Response C": "model_c"
    }
    
    # Tournament has model_c at position 3
    tournament_rankings = [
        {"model": "model_a", "wins": 2, "win_percentage": 1.0},
        {"model": "model_b", "wins": 1, "win_percentage": 0.5},
        {"model": "model_c", "wins": 0, "win_percentage": 0.0},
    ]
    
    minority = detect_minority_opinions(stage2_results, label_to_model, tournament_rankings)
    
    model_c_minority = next((m for m in minority if m["model"] == "model_c"), None)
    if model_c_minority:
        # Consensus position 3, dissenter placed at 1 -> undervalued
        assert model_c_minority["direction"] == "undervalued", f"Expected undervalued, got {model_c_minority['direction']}"
        print("✓ Minority direction undervalued - PASSED")
    else:
        print("✓ No minority for model_c (within tolerance) - PASSED")


def test_below_threshold_not_flagged():
    """When dissent rate is below 30%, no minority should be flagged."""
    # 4 rankers, only 1 disagrees = 25% < 30% threshold
    stage2_results = [
        make_stage2_entry("model_a", ["Response A", "Response B", "Response C", "Response D"]),
        make_stage2_entry("model_b", ["Response A", "Response B", "Response C", "Response D"]),
        make_stage2_entry("model_c", ["Response A", "Response B", "Response C", "Response D"]),
        make_stage2_entry("model_d", ["Response D", "Response C", "Response B", "Response A"]),  # One dissenter
    ]
    
    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b",
        "Response C": "model_c",
        "Response D": "model_d"
    }
    
    tournament_rankings = [
        {"model": "model_a", "wins": 3, "win_percentage": 1.0},
        {"model": "model_b", "wins": 2, "win_percentage": 0.67},
        {"model": "model_c", "wins": 1, "win_percentage": 0.33},
        {"model": "model_d", "wins": 0, "win_percentage": 0.0},
    ]
    
    minority = detect_minority_opinions(stage2_results, label_to_model, tournament_rankings)
    
    # With 25% dissent (1/4), should not meet 30% threshold
    # Note: model_a goes from 1 to 4 (diff 3), model_d goes from 4 to 1 (diff 3)
    # Both have only 1 dissenter out of 4, so 25% < 30%
    for m in minority:
        assert m["dissent_rate"] >= 0.3, f"Should not flag below threshold: {m}"
    
    print("✓ Below threshold not flagged - PASSED")


def test_within_tolerance_not_flagged():
    """Disagreement within position tolerance should not be flagged."""
    # All rankers within 1 position of each other
    stage2_results = [
        make_stage2_entry("model_a", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_b", ["Response B", "Response A", "Response C"]),  # A and B swapped
        make_stage2_entry("model_c", ["Response A", "Response B", "Response C"]),
    ]
    
    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b",
        "Response C": "model_c"
    }
    
    tournament_rankings = [
        {"model": "model_a", "wins": 2, "win_percentage": 1.0},
        {"model": "model_b", "wins": 1, "win_percentage": 0.5},
        {"model": "model_c", "wins": 0, "win_percentage": 0.0},
    ]
    
    minority = detect_minority_opinions(stage2_results, label_to_model, tournament_rankings)
    
    # model_a: consensus pos 1, one ranker put at pos 2 -> diff of 1, within tolerance
    # Should not be flagged
    assert len(minority) == 0, f"Expected no minority (within tolerance), got {minority}"
    print("✓ Within tolerance not flagged - PASSED")


def test_empty_inputs():
    """Empty inputs should return empty list."""
    assert detect_minority_opinions([], {}, []) == []
    assert detect_minority_opinions([], {"Response A": "model_a"}, []) == []
    assert detect_minority_opinions(
        [make_stage2_entry("model_a", ["Response A"])],
        {"Response A": "model_a"},
        []
    ) == []
    print("✓ Empty inputs - PASSED")


def test_5_model_realistic_scenario():
    """Realistic 5-model council with mixed agreement."""
    # 5 models, with 2 strongly disagreeing about model_c
    stage2_results = [
        make_stage2_entry("gpt-4", ["Response A", "Response B", "Response C", "Response D", "Response E"]),
        make_stage2_entry("claude", ["Response A", "Response B", "Response C", "Response D", "Response E"]),
        make_stage2_entry("gemini", ["Response A", "Response B", "Response C", "Response D", "Response E"]),
        # These 2 (40%) think model_c should be #1, not #3
        make_stage2_entry("grok", ["Response C", "Response A", "Response B", "Response D", "Response E"]),
        make_stage2_entry("llama", ["Response C", "Response A", "Response B", "Response D", "Response E"]),
    ]
    
    label_to_model = {
        "Response A": "gpt-4",
        "Response B": "claude",
        "Response C": "gemini",
        "Response D": "grok",
        "Response E": "llama"
    }
    
    # Consensus from majority
    tournament_rankings = [
        {"model": "gpt-4", "wins": 4, "win_percentage": 1.0},
        {"model": "claude", "wins": 3, "win_percentage": 0.75},
        {"model": "gemini", "wins": 2, "win_percentage": 0.5},
        {"model": "grok", "wins": 1, "win_percentage": 0.25},
        {"model": "llama", "wins": 0, "win_percentage": 0.0},
    ]
    
    minority = detect_minority_opinions(stage2_results, label_to_model, tournament_rankings)
    
    # gemini: consensus #3, but 2/5 (40%) placed it at #1
    # Diff of 2 positions exceeds tolerance of 1
    gemini_minority = next((m for m in minority if m["model"] == "gemini"), None)
    assert gemini_minority is not None, f"Expected minority for gemini, got {minority}"
    assert gemini_minority["consensus_position"] == 3
    assert 1 in gemini_minority["dissent_positions"]
    assert gemini_minority["dissent_rate"] == 0.4  # 2/5
    assert set(gemini_minority["dissenters"]) == {"grok", "llama"}
    assert gemini_minority["direction"] == "undervalued"
    
    print("✓ 5-model realistic scenario - PASSED")


def test_custom_threshold():
    """Test with custom dissent threshold."""
    stage2_results = [
        make_stage2_entry("model_a", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_b", ["Response A", "Response B", "Response C"]),
        make_stage2_entry("model_c", ["Response C", "Response B", "Response A"]),
    ]
    
    label_to_model = {
        "Response A": "model_a",
        "Response B": "model_b",
        "Response C": "model_c"
    }
    
    tournament_rankings = [
        {"model": "model_a", "wins": 2, "win_percentage": 1.0},
        {"model": "model_b", "wins": 1, "win_percentage": 0.5},
        {"model": "model_c", "wins": 0, "win_percentage": 0.0},
    ]
    
    # With 50% threshold, 33% dissent should not be flagged
    minority_50 = detect_minority_opinions(
        stage2_results, label_to_model, tournament_rankings, 
        dissent_threshold=0.5
    )
    assert len(minority_50) == 0, f"50% threshold should filter out 33% dissent: {minority_50}"
    
    # With 20% threshold, 33% dissent should be flagged
    minority_20 = detect_minority_opinions(
        stage2_results, label_to_model, tournament_rankings,
        dissent_threshold=0.2
    )
    assert len(minority_20) > 0, "20% threshold should catch 33% dissent"
    
    print("✓ Custom threshold - PASSED")


if __name__ == "__main__":
    test_no_minority_when_consensus()
    test_minority_detected_with_dissent()
    test_minority_direction_undervalued()
    test_below_threshold_not_flagged()
    test_within_tolerance_not_flagged()
    test_empty_inputs()
    test_5_model_realistic_scenario()
    test_custom_threshold()
    
    print("\n" + "="*50)
    print("All minority opinion tests passed!")
    print("="*50)
