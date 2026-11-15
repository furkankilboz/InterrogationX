"""Manually recorded behavioral observations (emotion, body language, gestures, facial expression).

Observations are typed in by the interviewer; nothing here is detected from images, audio or video. They are
descriptive notes. Nonverbal cues are weak and unreliable indicators of deception (DePaulo et al., 2003; Bond &
DePaulo, 2006; Vrij, Hartwig & Granhag, 2019), so this module never turns them into a truthfulness score. It only
compares each turn with the person's own baseline and shows where observations coincide with contested answers.
"""
import json
import re

import config

CHANNELS = {
    "facial": "Facial expression",
    "eyes": "Eyes and gaze",
    "hands": "Hands and gestures",
    "posture": "Posture and body",
    "legs": "Legs and feet",
    "voice": "Voice and speech",
    "emotion": "Displayed emotion",
    "other": "Other",
}
INTENSITIES = {1: "slight", 2: "moderate", 3: "strong"}
# Same label set as the text emotion classifier, so observed and text emotion can be shown side by side.
EMOTIONS = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Word prefix -> channel, checked in this order; the first channel with a matching word wins.
_CHANNEL_KEYWORDS = [
    ("legs", ("leg", "knee", "foot", "feet", "toe", "heel")),
    ("emotion", ("tear", "cry", "cries", "crying", "sob", "weep", "angry", "furious", "afraid", "scared", "panic")),
    ("eyes", ("eye", "gaze", "glance", "blink", "stare", "staring", "look", "squint")),
    ("voice", ("voice", "whisper", "pause", "silence", "silent", "laugh", "sigh", "swallow", "throat", "stammer",
               "stutter", "trail", "mutter", "mumbl", "breath", "exhale", "inhale", "quiet", "tone", "loud",
               "shout", "gulp", "chuckle", "audible", "speak", "hesitat")),
    ("facial", ("smile", "smirk", "frown", "jaw", "lip", "face", "brow", "expression", "microexpression",
                "grimace", "blush", "pale", "wince", "flush", "nostril", "cheek")),
    ("hands", ("hand", "finger", "nail", "fist", "palm", "wring", "fidget", "rub", "grip", "clench", "gestur",
               "glasses", "hair", "sleeve", "pick", "tap", "wipe", "scratch", "thumb", "knuckle", "pen", "neck")),
    ("posture", ("lean", "slouch", "shift", "sit", "stand", "shoulder", "arm", "shrug", "stiff", "straighten",
                 "rigid", "posture", "freez", "froze", "nod", "head", "upright", "slump", "chair", "back")),
]
_CHANNEL_PATTERNS = [(channel, re.compile(r"\b(?:" + "|".join(keywords) + ")"))
                     for channel, keywords in _CHANNEL_KEYWORDS]


def classify_cue(text):
    """Guess the channel of a free-text cue such as "picks at fingernails" (-> "hands")."""
    lowered = text.lower()
    for channel, pattern in _CHANNEL_PATTERNS:
        if pattern.search(lowered):
            return channel
    return "other"


def make_observation(channel, cue, intensity=2, note=""):
    if channel not in CHANNELS:
        raise ValueError(f"Unknown channel {channel!r}; use one of {', '.join(CHANNELS)}")
    if intensity not in INTENSITIES:
        raise ValueError(f"Intensity must be one of {sorted(INTENSITIES)}")
    cue = " ".join(cue.split()).lower()
    if not cue:
        raise ValueError("An observation needs a cue")
    return {"channel": channel, "cue": cue, "intensity": intensity, "note": note.strip()}


def load_cue_vocabulary(path=config.CUE_VOCABULARY):
    """Suggested cues per channel for the entry form; free text is always allowed too."""
    with open(path, encoding="utf-8") as f:
        vocabulary = json.load(f)["channels"]
    vocabulary["emotion"] = sorted(set(vocabulary.get("emotion", [])) | set(EMOTIONS))
    return {channel: vocabulary.get(channel, []) for channel in CHANNELS}


def baseline_cues(turns, baseline_turns):
    """Cues seen per channel in the first `baseline_turns` turns (the person's own normal behavior)."""
    baseline = {}
    for turn in turns[:baseline_turns]:
        for observation in turn.get("observations", []):
            baseline.setdefault(observation["channel"], set()).add(observation["cue"])
    return baseline


def baseline_deviations(turn, baseline):
    """Observations that differ from the baseline of their channel. A channel never observed during the
    baseline turns cannot be compared, so its observations are not counted as deviations."""
    return [observation for observation in turn.get("observations", [])
            if observation["channel"] in baseline and observation["cue"] not in baseline[observation["channel"]]]


def summarize_behavior(turn_analyses, baseline_turns):
    """Counts for the report. Every turn analysis has 'observations', 'deviations' and 'contested'."""
    channel_counts = {}
    for turn in turn_analyses:
        for observation in turn["observations"]:
            channel = observation["channel"]
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

    # Deviations from the baseline, separately for contested and uncontested turns after the baseline.
    deviations = {"contested": {"turns": 0, "deviations": 0}, "uncontested": {"turns": 0, "deviations": 0}}
    for turn in turn_analyses[baseline_turns:]:
        group = deviations["contested"] if turn["contested"] else deviations["uncontested"]
        group["turns"] += 1
        group["deviations"] += len(turn["deviations"])
    for group in deviations.values():
        group["per_turn"] = group["deviations"] / group["turns"] if group["turns"] else 0.0

    return {
        "total_observations": sum(channel_counts.values()),
        "channel_counts": channel_counts,
        "baseline_turns": baseline_turns,
        "deviations": deviations,
    }
