from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.flags import (
    event_pipeline_mode,
    event_pipeline_publishes,
    event_pipeline_selected_for,
    legacy_pipeline_publishes,
    legacy_pipeline_publishes_for,
    rollout_bucket,
    rollout_percent,
    rollout_phase_config,
)


def test_rollout_percent_zero_disables_publishing(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "event_pipeline")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 0)
    assert event_pipeline_mode() == "event_pipeline"
    assert not event_pipeline_publishes()
    assert not event_pipeline_selected_for("user-1")
    assert legacy_pipeline_publishes_for("user-1")


def test_legacy_mode_is_instant_rollback(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "legacy")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100)
    assert event_pipeline_mode() == "legacy"
    assert not event_pipeline_publishes()
    assert not event_pipeline_selected_for("user-1")
    assert legacy_pipeline_publishes()


def test_canary_selection_is_deterministic(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "event_pipeline")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 5)
    user = "stable-user-42"
    first = event_pipeline_selected_for(user)
    second = event_pipeline_selected_for(user)
    assert first is second
    assert rollout_bucket(user) == rollout_bucket(user)


def test_full_rollout_selects_everyone(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "event_pipeline")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100)
    users = ["user A", "user B", "user C", "anyone", "user-uuid-1", "session-zz", ""]
    for user in users:
        assert event_pipeline_selected_for(user), user
        assert not legacy_pipeline_publishes_for(user), user
    assert event_pipeline_publishes()
    assert not legacy_pipeline_publishes()
    assert event_pipeline_selected_for(None, None)
    assert event_pipeline_selected_for(None, "session-only")


def test_legacy_mode_wins_regardless_of_rollout_percent(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "legacy")
    for percent in (0, 5, 50, 100):
        monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", percent)
        assert event_pipeline_mode() == "legacy"
        assert not event_pipeline_publishes()
        assert not event_pipeline_selected_for("user A")
        assert not event_pipeline_selected_for("user B")
        assert not event_pipeline_selected_for("user C")
        assert legacy_pipeline_publishes()


def test_rollout_percent_zero_does_not_publish_event_pipeline_for_any_user(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "event_pipeline")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 0)
    for user in ("user A", "user B", "user C", "arbitrary-9"):
        assert not event_pipeline_selected_for(user)
        assert legacy_pipeline_publishes_for(user)
    assert not event_pipeline_publishes()


def test_phase_configs_are_documented_not_auto_advanced():
    assert rollout_phase_config(0)["mode"] == "shadow"
    assert rollout_phase_config(1)["percent"] == 5
    assert rollout_phase_config(2)["percent"] == 10
    assert rollout_phase_config(3)["percent"] == 25
    assert rollout_phase_config(4)["percent"] == 50
    assert rollout_phase_config(5)["percent"] == 100


def test_python_hash_is_not_used_for_rollout():
    left = rollout_bucket("user-7")
    right = rollout_bucket("user-7")
    assert left == right
    assert 0 <= left < 100
    assert rollout_percent() >= 0
