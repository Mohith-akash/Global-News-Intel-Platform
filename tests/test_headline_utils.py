"""Unit tests for src.headline_utils — URL headline extraction and cleaning."""

import importlib.util
from pathlib import Path

# Load the module file directly: headline_utils is stdlib-only, but importing
# it through the src package would pull in pandas/duckdb via src/__init__.py.
_spec = importlib.util.spec_from_file_location(
    "headline_utils", Path(__file__).resolve().parent.parent / "src" / "headline_utils.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean_headline = _mod.clean_headline
dedupe_headlines_simple = _mod.dedupe_headlines_simple
extract_headline_from_url = _mod.extract_headline_from_url
get_best_headline = _mod.get_best_headline
score_headline_quality = _mod.score_headline_quality


class TestExtractHeadlineFromUrl:
    def test_rejects_empty_and_non_string_input(self):
        assert extract_headline_from_url("") is None
        assert extract_headline_from_url(None) is None

    def test_rejects_url_with_no_path(self):
        assert extract_headline_from_url("https://www.reuters.com/") is None

    def test_extracts_slug_from_article_url(self):
        url = "https://www.bbc.com/news/world-europe-germany-approves-new-energy-plan"
        result = extract_headline_from_url(url)
        assert result is not None
        assert "germany" in result.lower()

    def test_skips_generic_segments(self):
        url = "https://example.com/news/article/parliament-votes-on-climate-bill-today"
        result = extract_headline_from_url(url)
        assert result is not None
        assert "parliament" in result.lower()

    def test_rejects_uuid_only_path(self):
        url = "https://example.com/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert extract_headline_from_url(url) is None

    def test_strips_html_extension(self):
        url = "https://example.com/chancellor-announces-major-tax-reform-package.html"
        result = extract_headline_from_url(url)
        assert result is not None
        assert ".html" not in result


class TestCleanHeadline:
    def test_rejects_empty_and_non_string_input(self):
        assert clean_headline("") is None
        assert clean_headline(None) is None

    def test_rejects_too_short_text(self):
        assert clean_headline("short text") is None

    def test_title_cases_valid_headline(self):
        result = clean_headline("germany announces new climate policy for industrial sector")
        assert result is not None
        assert result[0].isupper()

    def test_keeps_small_words_lowercase(self):
        result = clean_headline("germany announces the new climate policy for industry leaders")
        assert result is not None
        assert " the " in result

    def test_rejects_generic_headline(self):
        assert clean_headline("breaking news about something happening somewhere today") is None

    def test_strips_embedded_timestamps(self):
        result = clean_headline("parliament passes controversial 20260105 budget legislation for next year")
        if result is not None:
            assert "20260105" not in result

    def test_drops_truncated_trailing_modal_verb(self):
        result = clean_headline("president says the military intervention in region could")
        if result is not None:
            assert not result.lower().endswith("could")


class TestScoreHeadlineQuality:
    def test_empty_headline_scores_zero(self):
        assert score_headline_quality("", "https://reuters.com/x") == 0.0
        assert score_headline_quality(None, None) == 0.0

    def test_score_is_bounded(self):
        headlines = [
            "Germany Approves New Energy Plan for 2026",
            "Bnzqvyw5 Xes7 Tiqkizlx Qwrtplk Zzzqq",
            "LIVE UPDATES BREAKING 123456789 NEWS TICKER",
        ]
        for h in headlines:
            score = score_headline_quality(h, "https://example.com/a")
            assert 0.0 <= score <= 1.0

    def test_trusted_domain_scores_higher(self):
        headline = "Germany Approves New Energy Plan for Industry"
        trusted = score_headline_quality(headline, "https://www.reuters.com/a")
        unknown = score_headline_quality(headline, "https://randomblog.example/a")
        assert trusted > unknown

    def test_garbage_words_penalized(self):
        good = score_headline_quality(
            "Germany Approves New Energy Plan for Industry", "https://example.com/a"
        )
        garbage = score_headline_quality(
            "Bnzqvyw5 Xes7 Tiqkizlx Approves Plan", "https://example.com/a"
        )
        assert good > garbage


class TestDedupeHeadlinesSimple:
    def test_keeps_unique_headlines(self):
        headlines = [
            "Germany approves new energy plan",
            "France elects new prime minister today",
        ]
        assert dedupe_headlines_simple(headlines) == [0, 1]

    def test_drops_near_duplicates_differing_in_first_word(self):
        headlines = [
            "Germany approves new energy plan today",
            "Berlin approves new energy plan today",
        ]
        assert dedupe_headlines_simple(headlines) == [0]

    def test_skips_empty_entries(self):
        headlines = ["", None, "Germany approves new energy plan"]
        assert dedupe_headlines_simple(headlines) == [2]


class TestGetBestHeadline:
    def test_prefers_valid_db_headline(self):
        result = get_best_headline(
            "germany announces new climate policy for industrial sector",
            "https://example.com/some-other-completely-different-story-here",
        )
        assert result is not None
        assert "climate" in result.lower()

    def test_falls_back_to_url_extraction(self):
        result = get_best_headline(
            "", "https://example.com/parliament-votes-on-major-climate-bill-this-week"
        )
        assert result is not None
        assert "parliament" in result.lower()

    def test_returns_none_when_nothing_usable(self):
        assert get_best_headline("", "https://example.com/") is None
