import pytest

from observations import (CHANNELS, EMOTIONS, baseline_cues, baseline_deviations, classify_cue, load_cue_vocabulary,
                          make_observation, summarize_behavior)


@pytest.mark.parametrize("cue, channel", [
    ("picks at fingernails", "hands"),
    ("bounces knee rapidly", "legs"),
    ("looks away", "eyes"),
    ("voice cracking", "voice"),
    ("jaw clenches", "facial"),
    ("open expression", "facial"),   # "pen" inside "open" must not make this a hand cue
    ("crosses arms", "posture"),
    ("sobbing", "emotion"),
    ("stops", "other"),
])
def test_classify_cue(cue, channel):
    assert classify_cue(cue) == channel


def test_make_observation_normalizes_and_validates():
    assert make_observation("hands", "  Fidgeting   with Sleeve ", 3, " left hand ") == {
        "channel": "hands", "cue": "fidgeting with sleeve", "intensity": 3, "note": "left hand"}
    with pytest.raises(ValueError):
        make_observation("aura", "glowing")
    with pytest.raises(ValueError):
        make_observation("hands", "fidgeting", intensity=7)
    with pytest.raises(ValueError):
        make_observation("hands", "   ")


def test_cue_vocabulary_covers_every_channel_without_labels():
    vocabulary = load_cue_vocabulary()
    assert set(vocabulary) == set(CHANNELS)
    assert vocabulary["emotion"] == sorted(EMOTIONS)
    assert all(vocabulary[channel] for channel in ("facial", "eyes", "hands", "posture", "legs", "voice"))


def test_baseline_and_deviations():
    turns = [
        {"observations": [make_observation("posture", "upright"), make_observation("eyes", "steady eye contact")]},
        {"observations": [make_observation("posture", "leaning away"), make_observation("eyes", "steady eye contact"),
                          make_observation("hands", "fidgeting with sleeve")]},
    ]
    baseline = baseline_cues(turns, baseline_turns=1)
    assert baseline == {"posture": {"upright"}, "eyes": {"steady eye contact"}}
    # Hands were never observed during the baseline, so they cannot count as a deviation.
    assert [o["cue"] for o in baseline_deviations(turns[1], baseline)] == ["leaning away"]


def test_summarize_behavior_separates_contested_turns():
    observation = make_observation("posture", "leaning away")
    turns = [
        {"observations": [make_observation("posture", "upright")], "deviations": [], "contested": False},
        {"observations": [observation], "deviations": [observation], "contested": True},
        {"observations": [], "deviations": [], "contested": False},
    ]
    summary = summarize_behavior(turns, baseline_turns=1)
    assert summary["total_observations"] == 2 and summary["channel_counts"] == {"posture": 2}
    assert summary["deviations"]["contested"] == {"turns": 1, "deviations": 1, "per_turn": 1.0}
    assert summary["deviations"]["uncontested"] == {"turns": 1, "deviations": 0, "per_turn": 0.0}
