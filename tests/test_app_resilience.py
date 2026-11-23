"""A failing tab must not blank the rest of the page."""

from streamlit.testing.v1 import AppTest


def _page():
    import streamlit as st
    from components.ui import section
    from src.database import WarehouseUnavailable

    def broken():
        raise KeyError("confidence")

    def offline():
        raise WarehouseUnavailable("down")

    section(broken)
    section(offline)
    section(st.write, "still here")


def test_failing_sections_stay_contained():
    at = AppTest.from_function(_page).run()
    assert not at.exception
    assert [e.value for e in at.error] == ["This section couldn't load. The other tabs still work."]
    assert len(at.warning) == 1
    assert any("still here" in m.value for m in at.markdown)
