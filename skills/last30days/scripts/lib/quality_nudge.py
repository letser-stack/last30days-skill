"""Post-research quality score and upgrade nudge.

Computes a quality score based on the non-blocking core sources and builds
a nudge message describing what the user missed and how to fix it.

Fix text comes from ``lib.prescriptions`` (the single remediation
vocabulary shared with the doctor command, KTD 7); only the trigger
logic and the message framing live here.
"""

from typing import Any, List, Optional

from . import env, health, http, prescriptions, x_envelope


# Sources whose absence can justify a post-run quality repair. X remains a
# supported source, but it is optional: declining cookie access must not turn a
# successful multi-source run into a setup prompt or a lower quality grade.
CORE_SOURCES = ["hn", "polymarket", "x", "youtube", "reddit"]

# Labels for display
SOURCE_LABELS = {
    "hn": "Hacker News",
    "polymarket": "Polymarket",
    "x": "X/Twitter",
    "youtube": "YouTube",
    "reddit": "Reddit",
}


def _is_x_active(config: dict, research_results: dict) -> bool:
    """Check if X source is active (has credentials AND didn't error)."""
    if "x" in (research_results.get("active_sources") or []):
        return not bool(research_results.get("x_error"))
    has_creds = _has_x_credentials(config)
    if not has_creds:
        return False
    # If X errored this run, it's configured but broken
    if research_results.get("x_error"):
        return False
    return True


def _has_x_credentials(config: dict) -> bool:
    """Return True when any X/Twitter source credential is configured.

    ``X_BEARER_TOKEN`` counts where the xapi backend can actually run: an
    official-only host (it heads the auto chain there) or an explicit xapi
    pin. An ambient bearer on any other host is not a configured X source,
    so nothing changes there.
    """
    if (
        config.get("AUTH_TOKEN")
        or config.get("XAI_API_KEY")
        or config.get("XQUIK_API_KEY")
    ):
        return True
    if config.get("X_BEARER_TOKEN"):
        return "xapi" in env.x_auto_chain(config) or env.x_backend_pin(config) == "xapi"
    return False


def _x_error_prescription(config: dict, research_results: dict) -> prescriptions.Prescription:
    """The fix for a configured X that errored, routed through the X policy.

    The pipeline stamps the failed backend into ``x_error`` or
    ``x_degraded_error``, sometimes behind a simplified-query retry label.
    Use that runtime provenance before the host policy.
    """
    message = str(research_results.get("x_error") or research_results.get("x_degraded_error") or "")
    if message.startswith(x_envelope.DETAIL_NOT_PASSED):
        # The model declared the X connector lane and passed no envelope:
        # the fix is the connector, on any host.
        return prescriptions.for_x(config, "connector_missing")
    backend_error = message.removeprefix("Simplified-query retry failed: ")
    backend_error = backend_error.removeprefix("All X backends failed — ")
    if backend_error.startswith("X served via "):
        served_backend, separator, origin_error = backend_error.removeprefix("X served via ").partition(" after ")
        if separator:
            backend_error = origin_error.split(f"; {served_backend}", 1)[0]
    if backend_error.startswith("xai:"):
        state = http.classify_failure(message=backend_error)
        if state == health.PAYMENT_REQUIRED:
            failure = "xai_payment_required"
        elif state == health.RATE_LIMITED:
            failure = "xai_rate_limited"
        elif state == health.TIMEOUT:
            failure = "xai_timeout"
        elif state == health.AUTH_FAILED or "model" in backend_error.lower():
            failure = "xai_error"
        else:
            failure = "xai_unavailable"
        return prescriptions.for_x(config, failure)
    failure = "cookies_expired"
    if env.x_policy(config).hint_namespace == "official":
        if http.classify_failure(message=message) == health.PAYMENT_REQUIRED:
            failure = "payment_required"
    return prescriptions.for_x(config, failure)


def _has_ytdlp() -> bool:
    """Return True when the local/free YouTube lane is available."""
    try:
        from . import youtube_yt
        return bool(youtube_yt.is_ytdlp_installed())
    except Exception:
        return False


def _youtube_returned_data(research_results: dict) -> bool:
    """Return True when YouTube produced usable items through any provider."""
    videos = int(research_results.get("youtube_videos_count") or 0)
    transcripts = int(research_results.get("youtube_transcripts_count") or 0)
    return videos > 0 or transcripts > 0


def _is_youtube_active(config: dict, research_results: dict, *, has_ytdlp: bool) -> bool:
    """Check if YouTube source is active (yt-dlp installed)."""
    if not has_ytdlp:
        return False
    if research_results.get("youtube_error"):
        return False
    return True


# Below this transcript-fetch ratio, YouTube is considered "degraded" rather
# than active. Picked at 50% so a single legitimate caption-disabled video in a
# multi-video result does not trip the nudge, but a stale-yt-dlp run that fails
# every transcript does. Tunable via DEGRADED_TRANSCRIPT_THRESHOLD env var if
# operators need to adjust without code changes.
DEFAULT_DEGRADED_TRANSCRIPT_THRESHOLD = 0.5


def _is_youtube_degraded(research_results: dict, threshold: float) -> bool:
    """YouTube is degraded when videos were returned but the transcript-fetch
    ratio is below threshold. The canonical cause is a stale yt-dlp binary -
    YouTube's caption format changes frequently and old binaries silently fail
    every transcript while the search itself still succeeds.

    Captions-disabled videos are subtracted from the denominator: an uploader
    who turned off captions can never produce a transcript, so counting that
    video toward "fetch failures" produces false positives. A single
    captions-disabled video in a small result set was tripping the nudge.

    When actual fetch outcomes are available, they take precedence over the
    post-pruning ratio: the report counts only see items that survived
    freshness/relevance pruning, so a run where every transcript fetch
    succeeded but the fetched videos were later pruned looks identical to a
    stale-binary run (#531). Zero failures across attempted fetches proves
    the binary works - don't flag.
    """
    videos = int(research_results.get("youtube_videos_count") or 0)
    transcripts = int(research_results.get("youtube_transcripts_count") or 0)
    captions_disabled = int(research_results.get("youtube_captions_disabled_count") or 0)
    if videos <= 0:
        return False
    fetch_attempts = int(research_results.get("youtube_transcript_fetch_attempts") or 0)
    fetch_failures = int(research_results.get("youtube_transcript_fetch_failures") or 0)
    if fetch_attempts > 0 and fetch_failures == 0:
        return False
    eligible = videos - captions_disabled
    if eligible <= 0:
        # Every returned video had captions disabled - upstream content fact,
        # not a yt-dlp problem. Don't flag.
        return False
    # The engine only tries for a few transcripts per search (the depth's
    # transcript target), so judge against that, not every surviving video:
    # 4 transcripts from 15 videos with a target of 2 is a healthy run.
    target = int(research_results.get("youtube_transcript_target") or 0)
    expected = min(target, eligible) if target > 0 else eligible
    return (transcripts / expected) < threshold


def _youtube_bot_check_text(research_results: dict, *, has_sc: bool) -> str:
    """The degraded-YouTube line when yt-dlp hit a YouTube bot check.

    A bot check is YouTube refusing this network, so upgrading yt-dlp cannot
    help; name the two fixes that do."""
    videos = int(research_results.get("youtube_videos_count") or 0)
    transcripts = int(research_results.get("youtube_transcripts_count") or 0)
    rescued = int(research_results.get("youtube_sc_rescued") or 0)
    if rescued:
        rescue = f"ScrapeCreators rescued {rescued} of them. "
    elif has_sc:
        rescue = "ScrapeCreators rescued none of them. "
    else:
        rescue = ""
    fixes = "route yt-dlp through a residential host with LAST30DAYS_YOUTUBE_SSH_HOST"
    if not has_sc:
        fixes += (
            ", or add SCRAPECREATORS_API_KEY for the ScrapeCreators transcript "
            "fallback"
        )
    return (
        f"YouTube returned {videos} videos and {transcripts} transcripts: "
        "YouTube bot-checked yt-dlp on this network. "
        f"{rescue}This is a bot check, not a stale yt-dlp, so upgrading it "
        f"won't help. To get more transcripts, {fixes}."
    )


def _instagram_bonus_outcome(config: dict, research_results: dict) -> str | None:
    """Return the bonus-source outcome for Instagram, or None when not applicable.

    Returns:
        ``"errored"``  — SC is configured, the source ran, and it returned an error
                         (e.g. HTTP 404 from the ScrapeCreators endpoint).
        ``"no_items"`` — SC is configured, the source ran with no error, but returned
                         zero items (endpoint was live but had nothing for the query).
        ``None``       — SC not configured, source excluded, or source not run this
                         invocation.
    """
    if not config.get("SCRAPECREATORS_API_KEY"):
        return None
    excluded = {
        s.strip().lower()
        for s in (config.get("EXCLUDE_SOURCES") or "").split(",")
        if s.strip()
    }
    included = {
        s.strip().lower()
        for s in (config.get("INCLUDE_SOURCES") or "").split(",")
        if s.strip()
    }
    if "instagram" in excluded or (included and "instagram" not in included):
        return None
    count = research_results.get("instagram_items_count")
    if count is None:
        return None  # source not run this invocation
    if int(count) == 0:
        if research_results.get("instagram_error"):
            return "errored"
        return "no_items"
    return None


_ENGINE_CORE = (
    ("hn", "hackernews"),
    ("polymarket", "polymarket"),
    ("reddit", "reddit"),
    ("x", "x"),
    ("youtube", "youtube"),
)
_RAN = frozenset({"ok_results", "ok_zero"})


def _run_row(executed: dict, engine_name: str) -> dict:
    row = executed.get(engine_name) or {}
    return row if isinstance(row, dict) else {}


def _score_from_run_status(
    executed: dict,
    config: dict,
    research_results: dict,
    *,
    has_ytdlp: bool,
    core_active: List[str],
    core_missing: List[str],
    core_errored: List[str],
    core_degraded: List[str],
) -> List[str]:
    """Count a core source only when this process searched it."""
    optional_omitted: List[str] = []
    x_configured = _has_x_credentials(config) or (
        "x" in (research_results.get("active_sources") or [])
    )
    for core_name, engine_name in _ENGINE_CORE:
        status = str(_run_row(executed, engine_name).get("status") or "not_run")
        if core_name == "x" and status == "not_run" and not x_configured:
            optional_omitted.append("x")
            continue
        if status in _RAN:
            core_active.append(core_name)
            if core_name == "x" and research_results.get("x_degraded_error"):
                core_degraded.append("x")
            if core_name == "youtube" and status == "ok_results":
                threshold = float(
                    config.get("DEGRADED_TRANSCRIPT_THRESHOLD")
                    or DEFAULT_DEGRADED_TRANSCRIPT_THRESHOLD
                )
                if _is_youtube_degraded(research_results, threshold):
                    core_degraded.append("youtube")
            continue
        core_missing.append(core_name)
        if status == "error":
            core_errored.append(core_name)
        elif core_name == "youtube" and has_ytdlp and research_results.get("youtube_error"):
            core_errored.append(core_name)
    return optional_omitted


def _score_assumed_core(
    config: dict,
    research_results: dict,
    *,
    has_ytdlp: bool,
    core_active: List[str],
    core_missing: List[str],
    core_errored: List[str],
    core_degraded: List[str],
) -> List[str]:
    """Legacy score for callers that have no per-source run record.

    HN, Polymarket, and Reddit stay assumed-active here so older unit tests
    keep their contract. The engine path passes ``source_run_status`` and does
    not use this branch.
    """
    core_active.append("hn")
    core_active.append("polymarket")
    core_active.append("reddit")
    optional_omitted: List[str] = []
    x_configured = _has_x_credentials(config) or (
        "x" in (research_results.get("active_sources") or [])
    )
    if _is_x_active(config, research_results):
        core_active.append("x")
        if research_results.get("x_degraded_error"):
            core_degraded.append("x")
    elif x_configured and research_results.get("x_error"):
        core_missing.append("x")
        core_errored.append("x")
    else:
        optional_omitted.append("x")

    yt_active = _is_youtube_active(config, research_results, has_ytdlp=has_ytdlp)
    youtube_returned_data = _youtube_returned_data(research_results)
    if yt_active:
        core_active.append("youtube")
        threshold = float(config.get("DEGRADED_TRANSCRIPT_THRESHOLD") or DEFAULT_DEGRADED_TRANSCRIPT_THRESHOLD)
        if _is_youtube_degraded(research_results, threshold):
            core_degraded.append("youtube")
    elif youtube_returned_data and not research_results.get("youtube_error"):
        core_active.append("youtube")
        core_degraded.append("youtube")
    else:
        core_missing.append("youtube")
        if has_ytdlp and research_results.get("youtube_error"):
            core_errored.append("youtube")
    return optional_omitted


def compute_quality_score(config: dict, research_results: dict) -> dict:
    """Compute research quality score from the non-blocking core sources.

    Args:
        config: Configuration dict from env.get_config()
        research_results: Dict with keys like x_error, youtube_error,
            reddit_error reflecting what happened this run. Optional keys
            ``youtube_videos_count`` and ``youtube_transcripts_count`` enable
            degraded-YouTube detection (transcript-fetch ratio below threshold,
            or fallback/provider data returned without local yt-dlp).
            Optional keys ``instagram_items_count`` and ``instagram_error``
            enable failed/no-items detection for the bonus Instagram source.
            ``x_degraded_error``
            carries a failed xAI lane when another backend served X items.

    Returns:
        {
            "score_pct": 0-100,
            "core_active": ["hn", "polymarket", ...],
            "core_missing": ["youtube"],
            "core_errored": [],          # configured but errored at top level
            "core_degraded": [],         # configured and returned items but quality below threshold
            "bonus_errored": [],         # bonus sources (Instagram, etc.) configured but failed or returned no items
            "nudge_text": "..." or None if all sources healthy
        }
    """
    core_active: List[str] = []
    core_missing: List[str] = []
    core_errored: List[str] = []
    core_degraded: List[str] = []
    bonus_errored: List[str] = []

    # When the run recorded source_run_status, a core source counts only if
    # this process searched it. The old path assumed HN, Polymarket, and
    # Reddit had run, which printed "4/5" after a plan that never called them.
    executed = research_results.get("source_run_status") or None
    has_ytdlp = _has_ytdlp()
    if executed:
        optional_omitted = _score_from_run_status(
            executed, config, research_results, has_ytdlp=has_ytdlp,
            core_active=core_active, core_missing=core_missing,
            core_errored=core_errored, core_degraded=core_degraded,
        )
    else:
        optional_omitted = _score_assumed_core(
            config, research_results, has_ytdlp=has_ytdlp,
            core_active=core_active, core_missing=core_missing,
            core_errored=core_errored, core_degraded=core_degraded,
        )

    # Bonus sources (Instagram, etc.): SC-key holders expect content but the
    # source can fail (HTTP 404) or return zero items without surfacing a
    # clear signal in the brief. Track both outcomes separately.
    _ig_outcome = _instagram_bonus_outcome(config, research_results)
    if _ig_outcome == "errored":
        bonus_errored.append("instagram:failed")
    elif _ig_outcome == "no_items":
        bonus_errored.append("instagram:no_items")

    scored_source_count = len(CORE_SOURCES) - len(optional_omitted)
    score_pct = int(len(core_active) / scored_source_count * 100)

    has_sc = bool(config.get("SCRAPECREATORS_API_KEY"))
    active_sources = research_results.get("active_sources") or []
    x_fix = _x_error_prescription(config, research_results) if "x" in core_errored or "x" in core_degraded else None
    nudge_text = _build_nudge_text(
        core_missing,
        core_errored,
        core_degraded,
        research_results,
        has_sc=has_sc,
        active_sources=active_sources,
        bonus_errored=bonus_errored,
        has_ytdlp=has_ytdlp,
        core_total=scored_source_count,
        x_fix=x_fix,
        suggest_youtube_comments=(
            not env.is_youtube_comments_available(
                config, ytdlp_installed=has_ytdlp
            )
            and "youtube_comments" not in env.exclude_sources(config)
        ),
    ) if (core_missing or core_degraded or bonus_errored) else None

    return {
        "score_pct": score_pct,
        "core_active": core_active,
        "core_missing": core_missing,
        "core_errored": core_errored,
        "core_degraded": core_degraded,
        "bonus_errored": bonus_errored,
        "nudge_text": nudge_text,
    }


def _build_nudge_text(
    core_missing: List[str],
    core_errored: List[str],
    core_degraded: List[str] = None,
    research_results: dict = None,
    has_sc: bool = False,
    active_sources: list = None,
    bonus_errored: List[str] = None,
    has_ytdlp: bool = False,
    core_total: int | None = None,
    x_fix: Optional[prescriptions.Prescription] = None,
    suggest_youtube_comments: bool = False,
) -> str:
    """Build human-readable nudge text describing what was missed or degraded.

    Prioritizes free suggestions. Optionally mentions bonus sources
    (TikTok, Instagram, Threads, Pinterest) if ScrapeCreators key is configured.
    YouTube comments are only listed when the opt-in can actually enable them:
    not already available through yt-dlp or INCLUDE_SOURCES, and not switched
    off with EXCLUDE_SOURCES.
    """
    lines: List[str] = []
    core_degraded = core_degraded or []
    bonus_errored = bonus_errored or []
    research_results = research_results or {}

    # Describe what was missed
    missed_parts: List[str] = []
    executed = research_results.get("source_run_status") or {}
    for src in core_missing:
        label = SOURCE_LABELS[src]
        engine_name = "hackernews" if src == "hn" else src
        row = executed.get(engine_name) if isinstance(executed, dict) else None
        status = str((row or {}).get("status") or "")
        reason = str((row or {}).get("reason") or "")
        if src in core_errored:
            suffix = f", {reason}" if reason else ""
            missed_parts.append(f"{label} (errored this run{suffix})")
        elif status:
            missed_parts.append(f"{label} ({status})")
        else:
            missed_parts.append(label)

    effective_total = core_total if core_total is not None else len(CORE_SOURCES)
    active_count = effective_total - len(core_missing)
    lines.append(f"Research quality: {active_count}/{effective_total} core sources.")
    if missed_parts:
        lines.append(f"Missing: {', '.join(missed_parts)}.")
    if core_degraded:
        degraded_labels = ", ".join(SOURCE_LABELS[s] for s in core_degraded)
        lines.append(f"Degraded: {degraded_labels}.")
    if bonus_errored:
        failed_sources = []
        unavailable_sources = []
        silent_sources = []
        for s in bonus_errored:
            if s == "instagram:failed":
                failed_sources.append("Instagram")
            elif s == "instagram:no_items":
                unavailable_sources.append("Instagram")
            else:
                silent_sources.append(s.capitalize())
        if failed_sources:
            lines.append(f"Bonus source failed: {', '.join(failed_sources)}.")
        if unavailable_sources:
            lines.append(f"Bonus source unavailable: {', '.join(unavailable_sources)}.")
        if silent_sources:
            lines.append(f"Bonus source silent: {', '.join(silent_sources)}.")
    lines.append("")

    # Free suggestions
    free_suggestions: List[str] = []

    # A configured X that errored is the only X entry that can reach
    # core_missing: unconfigured/declined X is an optional omission and never
    # lands here. Surface the repair instead of hiding the outage.
    if "x" in core_missing and "x" in core_errored:
        if x_fix is None:
            x_fix = prescriptions.get("x", "cookies_expired")
        free_suggestions.append(f"X/Twitter errored - {x_fix.fix_nl}.")

    if "x" in core_degraded:
        if x_fix is None:
            x_fix = prescriptions.get("x", "cookies_expired")
        free_suggestions.append(f"X/Twitter used a backup - {x_fix.fix_nl}.")

    if "youtube" in core_missing:
        yt_row = {}
        if isinstance(executed, dict):
            yt_row = executed.get("youtube") or {}
        yt_status = str(yt_row.get("status") or "")
        yt_reason = str(yt_row.get("reason") or "")
        if yt_status == "not_run":
            free_suggestions.append(
                "YouTube was not run. Add youtube to the --plan sources and "
                "to --search (--search replaces the source list; it does not add)."
            )
        elif yt_status == "error" and yt_reason == "banned":
            free_suggestions.append(
                "YouTube was blocked (bot check or HTTP 429). This is a ban, "
                "not a missing yt-dlp install, a timeout, or zero results."
            )
        elif yt_status == "error" and yt_reason == "timeout":
            free_suggestions.append(
                "YouTube timed out. This is a timeout, not a ban, a missing "
                "install, or zero results, so updating yt-dlp won't help."
            )
        elif yt_status == "unavailable":
            yt_fix = prescriptions.get("youtube", "ytdlp_missing")
            free_suggestions.append(
                "YouTube: video transcripts with key moments - often the deepest "
                f"explanations on any topic. Install yt-dlp: {yt_fix.fix_cli} (free)"
            )
        elif "youtube" in core_errored:
            yt_error = str(research_results.get("youtube_error") or "").strip()
            # The engine's own stream timeout says nothing about the binary's
            # age, so it must not prescribe an upgrade.
            if http.classify_failure(message=yt_error) == health.TIMEOUT:
                free_suggestions.append(
                    f"YouTube timed out ({yt_error}) - this is a timeout, "
                    "not a stale yt-dlp, so updating it won't help."
                )
            else:
                yt_fix = prescriptions.get("youtube", "ytdlp_stale")
                free_suggestions.append(
                    f"YouTube errored - update yt-dlp: {yt_fix.fix_cli}"
                )
        else:
            yt_fix = prescriptions.get("youtube", "ytdlp_missing")
            free_suggestions.append(
                "YouTube: video transcripts with key moments - often the deepest "
                f"explanations on any topic. Install yt-dlp: {yt_fix.fix_cli} (free)"
            )

    if "youtube" in core_degraded:
        videos = int(research_results.get("youtube_videos_count") or 0)
        transcripts = int(research_results.get("youtube_transcripts_count") or 0)
        captions_disabled = int(research_results.get("youtube_captions_disabled_count") or 0)
        if not has_ytdlp and _youtube_returned_data(research_results):
            install = prescriptions.get("youtube", "ytdlp_missing")
            # Tolerant lookup: alt_cli makes no arity promise, so an entry
            # gaining/losing a platform alternate must degrade the wording,
            # never crash the nudge path.
            scoop_install = install.alt_cli[0] if len(install.alt_cli) > 0 else install.fix_cli
            pip_install = install.alt_cli[1] if len(install.alt_cli) > 1 else scoop_install
            free_suggestions.append(
                f"YouTube returned {videos} videos and {transcripts} transcripts "
                "through a fallback/provider path, but local yt-dlp is not "
                "installed. Install yt-dlp to enable the free local YouTube lane "
                f"and reduce reliance on fallback providers: {install.fix_cli} "
                f"(macOS), {scoop_install} (Windows), or {pip_install}."
            )
        elif int(research_results.get("youtube_bot_checked") or 0) > 0:
            free_suggestions.append(
                _youtube_bot_check_text(research_results, has_sc=has_sc)
            )
        else:
            captions_note = ""
            if captions_disabled > 0:
                captions_note = (
                    f" ({captions_disabled} of those had captions disabled by the "
                    "uploader, which is a separate cause and not fixable on your end)"
                )
            update = prescriptions.get("youtube", "ytdlp_stale")
            # Same tolerant lookup as the install branch above.
            scoop_update = update.alt_cli[0] if len(update.alt_cli) > 0 else update.fix_cli
            pip_update = update.alt_cli[1] if len(update.alt_cli) > 1 else scoop_update
            free_suggestions.append(
                f"YouTube returned {videos} videos but only {transcripts} transcripts "
                f"captured{captions_note}. The most common remaining cause is a stale "
                "yt-dlp binary - YouTube's caption format changes frequently and old "
                "binaries silently fail every transcript. Update via your package "
                f"manager: {scoop_update} (Windows), {update.fix_cli} (macOS), "
                f"or {pip_update}."
            )

    if "instagram:failed" in bonus_errored or "instagram:no_items" in bonus_errored:
        free_suggestions.append(
            "Instagram returned 0 reels despite SC being configured. SC's "
            "v2 reels endpoint wraps Google Search and 500's frequently on "
            "multi-token queries. The skill now retries with hashtag-form "
            "automatically; if zero items still appear, the topic may have "
            "no reel coverage on Instagram. Try a single-word topic like "
            "the most distinctive noun in your query."
        )

    # Mention bonus opt-in sources when SC key is present
    if has_sc:
        bonus_hints = []
        if "threads" not in (active_sources or []):
            bonus_hints.append("Threads")
        if "pinterest" not in (active_sources or []):
            bonus_hints.append("Pinterest")
        if suggest_youtube_comments:
            bonus_hints.append("YouTube comments")
        if bonus_hints:
            free_suggestions.append(
                f"Your SC key also powers {', '.join(bonus_hints)}. "
                "Add them to INCLUDE_SOURCES in your .env to enable."
            )

    if free_suggestions:
        lines.append("Free fixes:")
        for s in free_suggestions:
            lines.append(f"  - {s}")
        lines.append("")

    # Bonus sources mention (non-blocking)
    if not has_sc:
        lines.append(
            "Bonus: TikTok and Instagram are available with a free "
            "ScrapeCreators key at scrapecreators.com (no affiliation)."
        )
    else:
        lines.append("last30days has no affiliation with any API provider.")

    return "\n".join(lines)


# Machine-readable per-source outcomes.
# Closed token set. ``reason`` is omitted on clean ok_* rows.
STATUSES = (
    "not_run",
    "unavailable",
    "error",
    "ok_zero",
    "ok_results",
    "blocked_missing_credential",
)

# Sources the daily coverage line has to account for even when a host plan
# forgets them. Engine ids, not the quality nudge's "hn" alias.
WATCH_SOURCES = (
    "x",
    "hackernews",
    "reddit",
    "polymarket",
    "github",
    "youtube",
    "tiktok",
    "instagram",
)

def _credential_blocked(source: str, config: dict[str, Any]) -> bool:
    """True when the adapter cannot search because its key is absent.

    TikTok's public helper treats an Apify token as availability, but the
    search adapter itself only calls ScrapeCreators. Either missing key is
    ``blocked_missing_credential`` until that adapter can use the token.
    """
    if source == "tiktok":
        return not bool(config.get("SCRAPECREATORS_API_KEY") or config.get("APIFY_API_TOKEN"))
    if source == "instagram":
        return not bool(config.get("SCRAPECREATORS_API_KEY"))
    return False


def _failure_reason(source: str, state: str, detail: str) -> str:
    text = detail.lower()
    if state == health.TIMEOUT or "timed out" in text or "timeout" in text:
        return "timeout"
    banned = state == health.RATE_LIMITED or any(
        marker in text
        for marker in ("429", "bot", "sign in to confirm", "too many requests")
    )
    if banned:
        return "banned" if source == "youtube" else "rate_limited"
    if state == health.AUTH_FAILED or "login required" in text:
        return "auth"
    return "error"


def _row(
    source: str,
    status: str,
    count: int,
    reason: str | None = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown source status {status}")
    row: dict[str, Any] = {
        "source": source,
        "status": status,
        "count": max(0, int(count)),
    }
    if reason:
        row["reason"] = reason
    return row


def _from_outcome(source: str, outcome: Any, count: int) -> dict[str, Any]:
    state = str(getattr(outcome, "state", "") or "")
    detail = str(getattr(outcome, "detail", "") or "")
    attempted = bool(getattr(outcome, "attempted", True))
    if not attempted or state == health.SKIPPED_UNCONFIGURED:
        if source == "youtube" and state == health.SKIPPED_UNCONFIGURED:
            return _row(source, "unavailable", count, "ytdlp_missing")
        return _row(source, "not_run", count, "not_scheduled")
    # Items mean the search returned evidence. A backup lane can still leave
    # a non-ok state (auth-failed on the primary); that stays in ``reason``
    # so the quality footer can keep the repair without pretending the
    # source never ran.
    reason = None
    if state not in (health.OK, health.NO_RESULTS):
        reason = _failure_reason(source, state, detail)
        if state == health.PARTIAL and reason == "error":
            reason = "partial"
    if count > 0:
        return _row(source, "ok_results", count, reason)
    if state in (health.OK, health.NO_RESULTS):
        return _row(source, "ok_zero", 0)
    return _row(source, "error", 0, reason or "error")


def _not_executed(
    source: str,
    *,
    config: dict[str, Any],
    named: set[str],
    search_list: list[str] | None,
    ytdlp_installed: bool,
) -> dict[str, Any]:
    if _credential_blocked(source, config):
        reason = "SCRAPECREATORS_API_KEY"
        if source == "tiktok" and not config.get("APIFY_API_TOKEN"):
            reason = "SCRAPECREATORS_API_KEY"
        return _row(source, "blocked_missing_credential", 0, reason)
    if source == "youtube" and not ytdlp_installed:
        return _row(source, "unavailable", 0, "ytdlp_missing")
    if search_list and source not in search_list:
        return _row(source, "not_run", 0, "excluded_by_search")
    if source not in named:
        return _row(source, "not_run", 0, "not_in_plan")
    return _row(source, "not_run", 0, "not_scheduled")


def build_source_run_status(
    *,
    named_sources: list[str] | None,
    search_list: list[str] | None,
    items_by_source: dict[str, list[Any]] | None,
    source_status: dict[str, Any] | None,
    config: dict[str, Any] | None,
    ytdlp_installed: bool | None = None,
) -> list[dict[str, Any]]:
    """One row per watched source, plus anything this run actually searched."""
    config = config or {}
    items_by_source = items_by_source or {}
    source_status = source_status or {}
    named = [source for source in (named_sources or []) if isinstance(source, str)]
    named_set = set(named)
    search = list(search_list) if search_list else None
    if ytdlp_installed is None:
        from . import youtube_yt
        ytdlp_installed = bool(youtube_yt.is_ytdlp_installed())

    order: list[str] = []
    for source in list(named) + list(WATCH_SOURCES) + list(items_by_source) + list(source_status):
        if source not in order:
            order.append(source)

    rows: list[dict[str, Any]] = []
    for source in order:
        count = len(items_by_source.get(source) or [])
        outcome = source_status.get(source)
        if outcome is not None or count:
            row = _from_outcome(source, outcome, count) if outcome is not None else _row(
                source, "ok_results" if count else "ok_zero", count,
            )
            # A credential wall that still produced an error string and no
            # items stays blocked, not a generic error.
            if (
                row["status"] in {"error", "not_run", "unavailable"}
                and count == 0
                and _credential_blocked(source, config)
            ):
                row = _row(source, "blocked_missing_credential", 0, "SCRAPECREATORS_API_KEY")
            rows.append(row)
            continue
        rows.append(
            _not_executed(
                source,
                config=config,
                named=named_set,
                search_list=search,
                ytdlp_installed=ytdlp_installed,
            )
        )
    return rows


def render_source_run_status(rows: list[dict[str, Any]]) -> list[str]:
    """One parseable stderr line per source. No free-text payload."""
    lines: list[str] = []
    for row in rows:
        parts = [
            f"source={row['source']}",
            f"status={row['status']}",
        ]
        reason = row.get("reason")
        if reason:
            parts.append(f"reason={reason}")
        parts.append(f"count={int(row.get('count') or 0)}")
        lines.append("[source-status] " + " ".join(parts))
    return lines


def as_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["source"]): row for row in rows}
