"""Explicit plans keep named sources under --quick, and coverage matches the run."""

from unittest.mock import patch

import pytest

from lib import instagram, planner, polymarket, pipeline, quality_nudge, schema, tiktok
from lib import health


def _plan(sources, intent="concept"):
    return {
        "intent": intent,
        "freshness_mode": "evergreen_ok",
        "cluster_mode": "none",
        "subqueries": [{
            "label": "primary",
            "search_query": "AI 3C glasses",
            "ranking_query": "What are people saying about AI glasses?",
            "sources": list(sources),
            "weight": 1.0,
        }],
    }


DAILY = ["x", "hackernews", "polymarket", "github"]
AVAILABLE = ["x", "hackernews", "polymarket", "github", "reddit", "youtube"]


def test_quick_host_plan_keeps_named_polymarket_and_github():
    plan = planner._sanitize_plan(
        _plan(DAILY), "AI glasses", AVAILABLE, DAILY, "quick",
        honor_plan_sources=True,
    )
    assert plan.subqueries[0].sources == DAILY


def test_quick_engine_plan_still_caps_at_two_sources():
    sqs = [schema.SubQuery(
        label="primary", search_query="q", ranking_query="q?",
        sources=list(AVAILABLE), weight=1.0,
    )]
    result = planner._trim_subqueries_for_depth(
        sqs, "concept", "quick", AVAILABLE, honor_plan_sources=False,
    )
    assert len(result[0].sources) == 2


def test_concept_exclusion_does_not_strip_a_named_polymarket_source():
    plan = planner._sanitize_plan(
        _plan(["polymarket", "hackernews"]), "AI glasses",
        ["polymarket", "hackernews"], None, "quick",
        honor_plan_sources=True,
    )
    assert plan.subqueries[0].sources == ["polymarket", "hackernews"]


def test_omitted_search_source_is_logged_and_not_scheduled(capsys):
    plan = planner._sanitize_plan(
        _plan(DAILY), "AI glasses", AVAILABLE, ["x", "hackernews"], "quick",
        honor_plan_sources=True,
    )
    assert plan.subqueries[0].sources == ["x", "hackernews"]
    err = capsys.readouterr().err
    assert "polymarket" in err and "github" in err
    assert "not scheduled" in err


def test_quick_pipeline_fetches_every_named_public_source():
    sources = ["hackernews", "polymarket", "github"]
    with patch.object(pipeline, "_retrieve_stream", return_value=([], {})) as retrieve:
        report = pipeline.run(
            topic="AI glasses", config={}, mock=True, web_backend="none",
            depth="quick", requested_sources=sources,
            external_plan=_plan(sources),
        )
    assert set(report.query_plan.subqueries[0].sources) >= set(sources)
    assert set(sources) <= {call.kwargs["source"] for call in retrieve.call_args_list}
    by_source = {row["source"]: row for row in report.artifacts["source_run_status"]}
    for source in sources:
        assert by_source[source]["status"] == "ok_zero"
        assert by_source[source]["count"] == 0
    assert by_source["tiktok"]["status"] == "blocked_missing_credential"
    assert by_source["instagram"]["status"] == "blocked_missing_credential"
    assert any(
        line.startswith("[source-status] source=polymarket status=ok_zero")
        for line in report.artifacts["source_run_status_lines"]
    )


def test_search_list_drop_is_excluded_by_search_not_a_silent_success():
    with patch.object(pipeline, "_retrieve_stream", return_value=([], {})):
        report = pipeline.run(
            topic="AI glasses", config={}, mock=True, web_backend="none",
            depth="quick", requested_sources=["hackernews"],
            external_plan=_plan(["hackernews", "polymarket", "github"]),
        )
    by_source = {row["source"]: row for row in report.artifacts["source_run_status"]}
    assert by_source["hackernews"]["status"] == "ok_zero"
    assert by_source["polymarket"]["status"] == "not_run"
    assert by_source["polymarket"]["reason"] == "excluded_by_search"
    assert by_source["github"]["reason"] == "excluded_by_search"
    quality = quality_nudge.compute_quality_score(
        {}, {"source_run_status": by_source, "active_sources": ["hackernews"]},
    )
    assert "polymarket" not in quality["core_active"]
    assert "reddit" not in quality["core_active"]
    assert "4/5" not in (quality["nudge_text"] or "")
    assert "not_run" in quality["nudge_text"]


def test_unrun_core_sources_are_not_counted_active():
    """The 2026-11-17 footer said 4/5 while Polymarket and Reddit never ran."""
    statuses = {
        "hackernews": {"source": "hackernews", "status": "ok_results", "count": 6},
        "x": {"source": "x", "status": "ok_results", "count": 12},
        "polymarket": {"source": "polymarket", "status": "not_run", "reason": "not_scheduled", "count": 0},
        "reddit": {"source": "reddit", "status": "not_run", "reason": "not_in_plan", "count": 0},
        "youtube": {"source": "youtube", "status": "unavailable", "reason": "ytdlp_missing", "count": 0},
        "github": {"source": "github", "status": "not_run", "reason": "not_scheduled", "count": 0},
        "tiktok": {"source": "tiktok", "status": "blocked_missing_credential", "reason": "SCRAPECREATORS_API_KEY", "count": 0},
        "instagram": {"source": "instagram", "status": "blocked_missing_credential", "reason": "SCRAPECREATORS_API_KEY", "count": 0},
    }
    quality = quality_nudge.compute_quality_score(
        {},
        {"source_run_status": statuses, "active_sources": ["x", "hackernews"]},
    )
    assert quality["core_active"] == ["hn", "x"]
    assert "polymarket" in quality["core_missing"]
    assert "reddit" in quality["core_missing"]
    assert quality["score_pct"] == 40
    assert "2/5" in quality["nudge_text"]
    assert "4/5" not in quality["nudge_text"]
    assert "Polymarket (not_run)" in quality["nudge_text"]
    assert "unavailable" in quality["nudge_text"]


@pytest.mark.parametrize("reason,status_token", [
    ("timeout", "timeout"),
    ("banned", "banned"),
])
def test_youtube_timeout_and_ban_stay_distinct(reason, status_token):
    state = health.TIMEOUT if reason == "timeout" else health.RATE_LIMITED
    outcome = schema.SourceOutcome(
        source="youtube", state=state, items_returned=0, attempted=True,
        detail="Search timed out after 30s" if reason == "timeout" else "HTTP Error 429",
    )
    rows = quality_nudge.build_source_run_status(
        named_sources=["youtube"],
        search_list=["youtube"],
        items_by_source={"youtube": []},
        source_status={"youtube": outcome},
        config={},
        ytdlp_installed=True,
    )
    youtube = next(row for row in rows if row["source"] == "youtube")
    assert youtube["status"] == "error"
    assert youtube["reason"] == status_token
    line = quality_nudge.render_source_run_status([youtube])[0]
    assert f"reason={status_token}" in line
    assert "count=0" in line


def test_youtube_zero_results_are_ok_zero_not_not_run():
    outcome = schema.SourceOutcome(
        source="youtube", state=health.NO_RESULTS, items_returned=0, attempted=True,
    )
    rows = quality_nudge.build_source_run_status(
        named_sources=["youtube"],
        search_list=["youtube"],
        items_by_source={},
        source_status={"youtube": outcome},
        config={},
        ytdlp_installed=True,
    )
    youtube = next(row for row in rows if row["source"] == "youtube")
    assert youtube["status"] == "ok_zero"
    assert "reason" not in youtube


def test_youtube_missing_binary_is_unavailable_without_a_search():
    rows = quality_nudge.build_source_run_status(
        named_sources=["youtube"],
        search_list=["youtube"],
        items_by_source={},
        source_status={},
        config={},
        ytdlp_installed=False,
    )
    youtube = next(row for row in rows if row["source"] == "youtube")
    assert youtube == {
        "source": "youtube",
        "status": "unavailable",
        "count": 0,
        "reason": "ytdlp_missing",
    }


def test_tiktok_and_instagram_without_keys_do_not_call_the_network():
    from lib import http as http_module

    with patch.object(http_module, "get") as http_get:
        tiktok_result = tiktok.search_tiktok("AI glasses", "2026-10-17", "2026-11-16", token=None)
        instagram_result = instagram.search_instagram(
            "AI glasses", "2026-10-17", "2026-11-16", token=None,
        )
    http_get.assert_not_called()
    assert "SCRAPECREATORS_API_KEY" in tiktok_result["error"]
    assert "SCRAPECREATORS_API_KEY" in instagram_result["error"]
    assert tiktok_result["items"] == []
    assert instagram_result["items"] == []


def test_tiktok_with_a_key_uses_the_adapter(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        return {"aweme_list": []}

    monkeypatch.setattr(tiktok.http, "get", fake_get)
    result = tiktok.search_tiktok(
        "AI glasses", "2026-10-17", "2026-11-16", token="test-key-not-real",
    )
    assert captured["url"].startswith("https://api.scrapecreators.com/")
    assert "test-key-not-real" not in captured["url"]
    assert result.get("error") != "No SCRAPECREATORS_API_KEY configured"


def _quick_youtube_results():
    return {
        "source_run_status": {
            "youtube": {"source": "youtube", "status": "ok_results", "count": 1},
            "hackernews": {"source": "hackernews", "status": "ok_results", "count": 6},
            "x": {"source": "x", "status": "ok_results", "count": 6},
            "polymarket": {"source": "polymarket", "status": "ok_zero", "count": 0},
            "github": {"source": "github", "status": "ok_zero", "count": 0},
            "reddit": {"source": "reddit", "status": "not_run", "reason": "not_in_plan", "count": 0},
            "tiktok": {"source": "tiktok", "status": "blocked_missing_credential", "count": 0},
            "instagram": {"source": "instagram", "status": "blocked_missing_credential", "count": 0},
        },
        "active_sources": ["youtube", "hackernews", "x", "polymarket", "github"],
        "youtube_videos_count": 1,
        "youtube_transcripts_count": 0,
        "youtube_transcript_fetch_attempts": 0,
        "youtube_transcript_fetch_failures": 0,
        "youtube_transcript_target": 0,
    }


def test_quick_transcript_skip_is_not_a_stale_ytdlp_nudge():
    """--quick sets the transcript target to 0. Zero captions is that skip."""
    results = _quick_youtube_results()
    results["youtube_transcripts_not_requested"] = True
    quality = quality_nudge.compute_quality_score({}, results)
    text = quality["nudge_text"] or ""
    assert "youtube" in quality["core_active"]
    assert "youtube" not in quality["core_degraded"]
    assert "Degraded: YouTube" not in text
    assert "stale" not in text.lower()
    assert "update yt-dlp" not in text.lower()
    assert quality_nudge._is_youtube_degraded(results, 0.5) is False


def test_failed_transcript_fetches_still_mark_youtube_degraded():
    """A real fetch that fails is not the quick skip and is not a version guess."""
    results = _quick_youtube_results()
    results["youtube_transcripts_not_requested"] = False
    results["youtube_transcript_fetch_attempts"] = 2
    results["youtube_transcript_fetch_failures"] = 2
    results["youtube_videos_count"] = 2
    results["youtube_transcript_target"] = 2
    quality = quality_nudge.compute_quality_score({}, results)
    text = quality["nudge_text"] or ""
    assert "youtube" in quality["core_degraded"]
    assert "Degraded: YouTube" in text
    assert "transcript" in text.lower()


def test_polymarket_module_is_the_public_search_used_by_the_pipeline():
    assert polymarket.search_polymarket.__module__ == "lib.polymarket"
