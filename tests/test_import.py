"""Test SAT Slack Bot."""

import sat_slack_bot


def test_import() -> None:
    """Test that the app can be imported."""
    assert isinstance(sat_slack_bot.__name__, str)
