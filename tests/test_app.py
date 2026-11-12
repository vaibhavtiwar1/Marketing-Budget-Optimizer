"""End-to-end smoke tests that run the Streamlit script headlessly."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

TIMEOUT = 120


def _app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def test_default_view_renders_plan():
    at = _app()
    labels = [m.label for m in at.metric]
    assert "Recommended spend" in labels
    assert "Expected leads" in labels
    assert not at.error


def test_target_mode_and_value_goal():
    at = _app()
    at.sidebar.radio[1].set_value("Hit a target").run()
    assert not at.exception
    at.sidebar.radio[2].set_value("Pipeline value").run()
    assert not at.exception
    assert "Pipeline value" in [m.label for m in at.metric]


def test_unreachable_target_shows_reason():
    at = _app()
    at.sidebar.radio[1].set_value("Hit a target").run()
    at.sidebar.number_input(key="target_leads").set_value(1_000_000).run()
    assert not at.exception
    assert any("isn't reachable" in e.value for e in at.error)


def test_save_scenario():
    at = _app()
    next(b for b in at.button if "Save scenario" in b.label).click().run()
    assert not at.exception
    assert len(at.session_state["scenarios"]) == 1
