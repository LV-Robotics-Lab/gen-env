"""Selection must use all frozen scenarios and an independent finer timestep."""

from self_improving.sim_adapters.genesis import tune_text_physics as tuning


def seed_results(passed=True):
    return {str(seed): {"passed": passed} for seed in tuning.SEEDS}


def test_three_base_seeds_are_not_a_sensitivity_check():
    assert tuning.choose_candidate({"a": seed_results()}, {}, ("a",)) is None


def test_all_six_results_must_pass():
    matrix = {"a": seed_results(), "b": seed_results()}
    half = {"a": seed_results(), "b": seed_results()}
    half["a"]["87"]["passed"] = False
    assert tuning.choose_candidate(matrix, half, ("a", "b")) == "b"
    del matrix["b"]["42"]
    assert tuning.choose_candidate(matrix, half, ("a", "b")) is None


def test_predeclared_preference_beats_result_order():
    matrix = {"b": seed_results(), "a": seed_results()}
    half = {"b": seed_results(), "a": seed_results()}
    assert tuning.choose_candidate(matrix, half, ("a", "b")) == "a"
