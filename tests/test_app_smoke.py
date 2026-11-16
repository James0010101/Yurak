from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ENTRYPOINT = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def test_app_loads_without_exceptions():
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=15)

    assert not at.exception, f"App crashed with exceptions: {at.exception}"


@pytest.mark.parametrize("page", ["app_pages/screening.py", "app_pages/methodology.py", "app_pages/about.py"])
def test_every_page_renders(page):
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=30)
    at.switch_page(page).run(timeout=60)
    assert not at.exception, f"{page} crashed: {at.exception}"


def test_screening_shows_heart_age_and_reacts_to_presets():
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=30)
    at.switch_page("app_pages/screening.py").run(timeout=60)
    labels = [m.label for m in at.metric]
    assert "Риск для сердца, 10 лет" in labels
    chd_before = next(m.value for m in at.metric if m.label == "Риск для сердца, 10 лет")

    at.button(key="btn_preset_a").click().run(timeout=60)
    assert not at.exception
    chd_after = next(m.value for m in at.metric if m.label == "Риск для сердца, 10 лет")
    assert chd_after != chd_before  # healthy profile gives a different (lower) risk


def test_prior_cvd_blocks_results():
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=30)
    at.switch_page("app_pages/screening.py").run(timeout=60)
    at.checkbox(key="cvd_0_mmol").check()
    at.button[-1].click()  # form submit sits after the preset buttons
    at.run(timeout=60)
    assert not at.exception
    assert not any(m.label == "Риск для сердца, 10 лет" for m in at.metric)


def test_age_outside_model_range_shows_markers_not_scores():
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=30)
    at.switch_page("app_pages/screening.py").run(timeout=60)
    at.number_input(key="age_0_mmol").set_value(25)
    at.button[-1].click()
    at.run(timeout=60)
    assert not at.exception
    assert not any(m.label == "Риск для сердца, 10 лет" for m in at.metric)
    assert len(at.dataframe) >= 1  # markers table is still shown


def test_switch_language_smoke():
    at = AppTest.from_file(ENTRYPOINT)
    at.run(timeout=15)


    # Check that selector exists and change to uz
    if at.selectbox:
        at.selectbox[0].select("uz").run(timeout=15)
        assert not at.exception

        # Switch to en
        at.selectbox[0].select("en").run(timeout=15)
        assert not at.exception
