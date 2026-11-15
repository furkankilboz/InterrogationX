import os

DB_FILE = os.environ.get("IXD_DB", "interrogation_x.db")
DATA_FILE = os.environ.get("IXD_DATA", "forensic_statements_data_en.json")
CHROMA_DIR = "./chroma_store"

# Classification thresholds
THRESH_CORROBORATION = 0.65
THRESH_PARTIAL = 0.35
THRESH_RELATED = 0.20
THRESH_CONTRA_TOPIC = 0.28

# Verdict thresholds
VERDICT_STRONG_CONTRA = 0.35
VERDICT_CONTRA = 0.18
VERDICT_STRONG_CORROB = 0.45
VERDICT_CORROB = 0.22

# Negation words
NEGATION_WORDS = frozenset({
    "not", "never", "no", "n't", "neither", "nor",
    "nobody", "nothing", "nowhere", "none", "cannot",
    "hardly", "barely", "scarcely", "without",
})

# Antonym pairs for contradiction detection
ANTONYM_PAIRS = {
    "fast": "slow",
    "slow": "fast",
    "loud": "quiet",
    "quiet": "loud",
    "angry": "calm",
    "calm": "angry",
    "early": "late",
    "late": "early",
    "inside": "outside",
    "outside": "inside",
    "left": "right",
    "right": "left",
    "enter": "exit",
    "exit": "enter",
    "stop": "continue",
    "continue": "stop",
    "arrive": "depart",
    "depart": "arrive",
    "alone": "together",
    "together": "alone",
    "sober": "drunk",
    "drunk": "sober",
    "armed": "unarmed",
    "unarmed": "armed",
    "guilty": "innocent",
    "innocent": "guilty",
    "present": "absent",
    "absent": "present",
    "awake": "asleep",
    "asleep": "awake",
    "aggressive": "passive",
    "passive": "aggressive",
    "running": "walking",
    "walking": "running",
    "push": "pull",
    "pull": "push",
    "steal": "return",
    "return": "steal",
    "attack": "defend",
    "defend": "attack",
    "shout": "whisper",
    "whisper": "shout",
    "hit": "miss",
    "miss": "hit",
    "polite": "rude",
    "rude": "polite",
    "safe": "dangerous",
    "dangerous": "safe",
    "open": "closed",
    "closed": "open",
    "friendly": "hostile",
    "hostile": "friendly",
    "honest": "dishonest",
    "dishonest": "honest",
    "willing": "reluctant",
    "reluctant": "willing",
    "visible": "hidden",
    "hidden": "visible",
    "intentional": "accidental",
    "accidental": "intentional",
    # Traffic / motion
    "yellow": "red",
    "red": "yellow",
    "green": "red",
    "stopped": "moving",
    "moving": "stopped",
    "speeding": "slow",
    "hurry": "calm",
    "reckless": "careful",
    "careful": "reckless",
    # Behavior
    "harmless": "threatening",
    "threatening": "harmless",
    "scared": "calm",
    "voluntary": "forced",
    "forced": "voluntary",
    "silent": "loud",
    "repeated": "once",
    "once": "repeated",
    "single": "multiple",
    "multiple": "single",
    # Emotional state
    "comfortable": "uncomfortable",
    "uncomfortable": "comfortable",
    "professional": "unprofessional",
    "unprofessional": "professional",
    "appropriate": "inappropriate",
    "inappropriate": "appropriate",
    # Physical actions
    "touched": "untouched",
    "untouched": "touched",
    "broke": "intact",
    "intact": "broke",
}

CONTRADICTING_VERB_PAIRS = frozenset({
    ("leave", "stay"),
    ("enter", "exit"),
    ("take", "return"),
    ("push", "pull"),
    ("shout", "whisper"),
    ("hit", "miss"),
    ("stop", "continue"),
    ("start", "end"),
    ("stop", "go"),
    ("stop", "enter"),
    ("brake", "accelerate"),
    ("wait", "rush"),
    ("arrive", "depart"),
    ("attack", "defend"),
    ("steal", "return"),
    ("run", "walk"),
    ("touch", "avoid"),
    ("see", "miss"),
    ("know", "ignore"),
    ("agree", "disagree"),
    ("deny", "admit"),
    ("accept", "reject"),
    ("approach", "flee"),
    ("grab", "release"),
    ("threaten", "reassure"),
    ("apologize", "insult"),
    ("help", "harm"),
    ("cooperate", "resist"),
    ("confess", "deny"),
    ("allow", "prevent"),
    ("remember", "forget"),
    ("sit", "stand"),
    ("climb", "descend"),
    ("raise", "lower"),
    ("pick", "drop"),
})

QUANTIFIER_OPPOSITES = {
    "all": {"none", "no"},
    "everyone": {"nobody", "no one", "noone"},
    "always": {"never"},
    "many": {"few", "none"},
    "most": {"few", "none"},
    "everything": {"nothing"},
    "everywhere": {"nowhere"},
    "several": {"none", "zero"},
    "often": {"rarely", "never", "seldom"},
    "completely": {"barely", "hardly"},
}

WORD_TO_NUMBER = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "thirty": 30,
    "hundred": 100,
}
