"""Datasets from the Hugging Face Hub. They are downloaded into external_data/ when needed (not committed).

  python external_datasets.py vocabulary   rebuild cue_vocabulary.json
  python external_datasets.py samples      rewrite the sample sessions in sessions/

Datasets:
  solarkyle/alibi-interrogation-dataset (Apache-2.0)
      1,736 synthetic interrogation dialogues written by a language model. The character card of each dialogue
      lists the true facts and says if the alibi is a lie. The replies contain stage directions such as
      "*picks at fingernails*"; here they become behavioral observations.
  darkknight25/Interrogation_Dataset_for_AI_Deception_Detection (MIT)
      1,600 synthetic question-answer rows with notes on face, hands, posture and legs. Used for the list of
      suggested cues and for the experiment in "benchmarks.py cues".
  pietrolesci/dialogue_nli (Dialogue NLI, Welleck et al. 2019)
      Sentence pairs checked by humans. Used as a contradiction benchmark. No license is given on the Hub,
      so nothing from it is copied into this repository.
"""
import json
import re
import sys
from collections import Counter

import config
from observations import CHANNELS, classify_cue, make_observation
from sessions import add_turn, new_session, save_session

# (repository on the Hub, file name)
ALIBI = ("solarkyle/alibi-interrogation-dataset", "alibi_training_data_clean.jsonl")
CUES = ("darkknight25/Interrogation_Dataset_for_AI_Deception_Detection",
        "secret_service_style_interogation_dataset.jsonl")
DNLI = ("pietrolesci/dialogue_nli", "data/verified_test-00000-of-00001.parquet")

_STAGE_DIRECTION = re.compile(r"\*([^*]{2,120})\*")
_NAME_LINE = re.compile(r"\*\*Name:\*\*\s*(.+)")
_GIVES_ANSWER_AWAY = re.compile(r"\b(innocent|guilty)\b", re.IGNORECASE)
_FLOOR_SUFFIX = re.compile(r"\s+on [a-z ]*(floor|rug|carpet)$")


def download(dataset):
    """Download one file of a dataset (or reuse the local copy) and return its path."""
    from huggingface_hub import hf_hub_download

    repo, filename = dataset
    folder = config.EXTERNAL_DATA_DIR / repo.replace("/", "__")
    return hf_hub_download(repo, filename, repo_type="dataset", local_dir=folder)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def card_sections(character_card):
    """Split a character card into its sections: {"Heading": ["bullet", "bullet", ...]}."""
    sections = {}
    current = None
    for line in character_card.splitlines():
        line = line.strip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current and line.startswith("- "):
            sections[current].append(line[2:].strip())
    return sections


def split_stage_directions(reply):
    """'*looks away* I was home.' -> ('I was home.', ['looks away'])."""
    directions = []
    for direction in _STAGE_DIRECTION.findall(reply):
        directions.append(" ".join(direction.split()))
    answer = " ".join(_STAGE_DIRECTION.sub(" ", reply).split())
    return answer, directions


def parse_alibi_conversation(conversation, index):
    """Turn one dialogue into a session.

    Returns None when the card has no name, no facts, or no clear "lie"/"truth" label for the alibi.
    session["ground_truth"] keeps that label for the benchmark; the analysis never reads it.
    """
    turns = conversation["conversations"]
    if not turns or turns[0]["from"] != "system":
        return None
    card = turns[0]["value"]
    name = _NAME_LINE.search(card)

    # Only the first "Secret Knowledge" section and the first "Alibi" section of a card are used.
    facts = None
    alibi_heading = None
    for heading, bullets in card_sections(card).items():
        if heading.startswith("Secret Knowledge") and facts is None:
            # Lines like "Dana is GUILTY" would give the answer away, so they are left out.
            facts = [bullet for bullet in bullets if not _GIVES_ANSWER_AWAY.search(bullet)]
        elif heading.startswith("Alibi") and alibi_heading is None:
            alibi_heading = heading
    label = None
    if alibi_heading and "(LIE)" in alibi_heading:
        label = "lie"
    elif alibi_heading and ("(TRUTH)" in alibi_heading or "(TRUE)" in alibi_heading):
        label = "truth"
    if not name or not facts or label is None:
        return None

    subject = name.group(1).strip()
    evidence = []
    for number, fact in enumerate(facts, start=1):
        evidence.append({"label": f"Fact {number}", "text": fact})
    session = new_session(f"alibi-{index:04d}", f"Interrogation of {subject}", subject=subject, baseline_turns=1,
                          evidence=evidence)
    session["source"] = f"{ALIBI[0]} (Apache-2.0), synthetic dialogue #{index}"
    session["ground_truth"] = {"alibi": label}

    question = ""
    for turn in turns[1:]:
        if turn["from"] == "human":
            question = turn["value"]
        elif turn["from"] == "gpt":
            answer, directions = split_stage_directions(turn["value"])
            if answer:
                observations = [make_observation(classify_cue(d), d) for d in directions]
                add_turn(session, question, answer, observations)
            question = ""
    if not session["turns"]:
        return None
    return session


def iter_alibi_sessions():
    for index, conversation in enumerate(read_jsonl(download(ALIBI))):
        session = parse_alibi_conversation(conversation, index)
        if session:
            yield session


def write_sample_sessions(per_label=3, min_turns=5, max_turns=8):
    """Save a few short sessions of each kind (false alibi, true alibi) for the app and the tests."""
    config.SESSIONS_DIR.mkdir(exist_ok=True)
    written = {"lie": 0, "truth": 0}
    for session in iter_alibi_sessions():
        label = session["ground_truth"]["alibi"]
        if written[label] < per_label and min_turns <= len(session["turns"]) <= max_turns:
            save_session(session, config.SESSIONS_DIR / f"{session['session_id']}.json")
            written[label] += 1
        if written["lie"] >= per_label and written["truth"] >= per_label:
            break
    return written


def build_cue_vocabulary(per_channel=14):
    """The most frequent cue phrases per channel, as suggestions for the observation form.

    The "culprit"/"genuine" tags of the source data are not used: a cue says nothing about truthfulness.
    """
    counts = {channel: Counter() for channel in CHANNELS}

    for row in read_jsonl(download(CUES)):
        for text in (row.get("cues") or {}).values():
            for phrase in text.split(","):
                phrase = " ".join(phrase.split()).lower()
                phrase = _FLOOR_SUFFIX.sub("", phrase)     # "tapping foot on tiled floor" -> "tapping foot"
                if len(phrase) > 4:
                    counts[classify_cue(phrase)][phrase] += 1

    for conversation in read_jsonl(download(ALIBI)):
        for turn in conversation["conversations"]:
            if turn["from"] == "gpt":
                answer, directions = split_stage_directions(turn["value"])
                for direction in directions:
                    counts[classify_cue(direction)][direction.lower()] += 1

    channels = {}
    for channel, counter in counts.items():
        if channel in ("other", "emotion"):
            continue
        channels[channel] = [cue for cue, count in counter.most_common(per_channel)]
    vocabulary = {
        "sources": [f"{CUES[0]} (MIT)", f"{ALIBI[0]} (Apache-2.0)"],
        "note": "Suggestions for the observation form only. A cue says nothing about truthfulness.",
        "channels": channels,
    }
    with open(config.CUE_VOCABULARY, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    return vocabulary


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "vocabulary":
        for channel, cues in build_cue_vocabulary()["channels"].items():
            print(f"{channel}: {', '.join(cues)}")
    elif command == "samples":
        print(f"Wrote sample sessions to {config.SESSIONS_DIR}: {write_sample_sessions()}")
    else:
        print(__doc__)
