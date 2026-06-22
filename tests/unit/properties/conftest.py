"""Package-level conftest for Hypothesis property-based tests.

Automatically applies the ``properties`` pytest marker to every test collected
from this package, and registers a deterministic Hypothesis CI profile so runs
are stable and reproducible in CI (no random seed variance).
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, Phase, settings

# ---------------------------------------------------------------------------
# Hypothesis profile: deterministic, stable for CI
# ---------------------------------------------------------------------------

settings.register_profile(
    "ci",
    derandomize=True,
    max_examples=50,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)
settings.load_profile("ci")


# ---------------------------------------------------------------------------
# Auto-marker: tag every test in this package with `properties`
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the ``properties`` marker to every test collected from this package.

    Args:
        items: All collected test items from the current pytest session.
    """
    pkg_prefix = str(__file__).removesuffix("conftest.py")
    marker = pytest.mark.properties
    for item in items:
        if str(item.fspath).startswith(pkg_prefix):
            item.add_marker(marker)
