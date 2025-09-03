# ch5_monte_carlo/tests/test_offpolicy_is.py
import numpy as np
from ch5_monte_carlo.examples.mc_offpolicy_is_demo import (
    TRUE_V, offpolicy_mc, monte_carlo_stats
)

def test_point_estimates_concentrate_near_truth():
    # With a moderately large N, both estimators should be close to 0.62
    N = 2000
    ois, wis = offpolicy_mc(N=N, seed=123)
    # Allow a small tolerance; OIS unbiased but noisy, WIS biased finite-N but tighter.
    assert abs(ois - TRUE_V) < 0.06, f"OIS too far from truth: {ois:.4f}"
    assert abs(wis - TRUE_V) < 0.06, f"WIS too far from truth: {wis:.4f}"

def test_wis_variance_lower_than_ois_variance():
    # Compare variance across multiple trials at modest N where the difference is visible
    stats = monte_carlo_stats(N=300, trials=400, seed=7)
    ois_var = stats["ois_var"]
    wis_var = stats["wis_var"]

    # WIS should have substantially lower variance than OIS in this toy setup.
    # We require ~40% reduction to avoid flaky CI.
    assert wis_var < 0.6 * ois_var, (
        f"WIS variance not sufficiently lower "
        f"(OIS var={ois_var:.5f}, WIS var={wis_var:.5f})"
    )

    # Means should be reasonably close to the true value in aggregate
    assert abs(stats["ois_mean"] - TRUE_V) < 0.04
    assert abs(stats["wis_mean"] - TRUE_V) < 0.04

