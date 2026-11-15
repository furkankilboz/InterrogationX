"""HTML report of an analyzed interrogation session (see interrogation_analysis.analyze_session).

The page is self-contained (inline CSS, no scripts) and uses simple table markup so it also renders in Qt's
QTextBrowser. Every piece of session text is escaped.
"""
from datetime import datetime
from html import escape

from observations import CHANNELS, INTENSITIES

_CSS = """
body { font-family: Segoe UI, Arial, sans-serif; color: #23272e; background: #ffffff; margin: 24px; font-size: 14px; }
h1 { font-size: 22px; margin-bottom: 2px; color: #1f2a37; }
h2 { font-size: 17px; margin-top: 28px; color: #1f2a37; border-bottom: 3px solid #d97706; padding-bottom: 4px; }
h3 { font-size: 15px; margin: 18px 0 4px 0; color: #0f766e; }
p.meta { color: #6b7280; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { border: 1px solid #d8d2c4; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #1f2a37; color: #ffffff; }
.notice { background: #fdf3dc; border: 1px solid #d97706; padding: 10px 12px; margin: 12px 0; }
.bad { color: #b42318; font-weight: bold; }
.good { color: #067647; font-weight: bold; }
.muted { color: #6b7280; }
.tag { background: #f3f0e9; border: 1px solid #d8d2c4; padding: 1px 6px; }
.question { color: #1f2a37; font-style: italic; }
"""

_STATUS_TEXT = {
    "contradicted": ("bad", "contradicted by the account"),
    "acknowledged": ("good", "acknowledged"),
    "not_addressed": ("muted", "not addressed"),
}


def _row(cells, header=False, css=""):
    tag = "th" if header else "td"
    attribute = f' class="{css}"' if css else ""
    return f"<tr{attribute}>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def _table(header, rows):
    """An HTML table. Pass header=None for a table without a header row."""
    html = "<table>"
    if header:
        html += _row(header, header=True)
    return html + "".join(rows) + "</table>"


def _turn_label(index):
    return f"Turn {index + 1}"


def _summary_section(analysis):
    s = analysis["summary"]
    rows = [_row([label, f"<b>{value}</b>"]) for label, value in [
        ("Question-answer turns", s["turns"]),
        ("Turns with a contested claim", s["contested_turns"]),
        ("Claims contradicted by the evidence", s["evidence_contradictions"]),
        ("Evidence items the account contradicts", s["evidence_items_contradicted"]),
        ("Claims contradicting an earlier answer", s["internal_contradictions"]),
        ("Turns with evasion cues", s["evasive_turns"]),
        ("Behavioral observations recorded", s["observations"]),
    ]]
    return "<h2>Summary</h2>" + _table(["Finding", "Count"], rows)


def _evidence_section(analysis):
    if not analysis["evidence"]:
        return "<h2>Evidence</h2><p class='muted'>No evidence was entered for this session.</p>"
    rows = []
    for item in analysis["evidence"]:
        css, text = _STATUS_TEXT[item["status"]]
        where = f" in {_turn_label(item['turn'])}" if item["turn"] is not None else ""
        rows.append(_row([escape(item["label"]), escape(item["text"]),
                          f"<span class='{css}'>{text}{where}</span>", f"{item['contradiction']:.2f}"]))
    index = analysis["summary"]["evidence_conflict_index"]
    return ("<h2>Evidence and how the account relates to it</h2>"
            + _table(["Item", "Text", "Status", "Contradiction score"], rows)
            + f"<p>Evidence conflict index: <b>{index:+.2f}</b> on a scale from -1 (the account acknowledges every "
              "item) to +1 (it contradicts every item). It summarizes the table above; it is not a measure of "
              "truthfulness.</p>")


def _contradiction_sections(analysis):
    parts = ["<h2>Statement-evidence contradictions</h2>"]
    if analysis["evidence_contradictions"]:
        parts.append(_table(["Turn", "Claim", "Conflicts most with", "Score"], [
            _row([_turn_label(c["turn"]), escape(c["claim"]), escape(c.get("evidence_label", "")),
                  f"{c['contradiction']:.2f}"]) for c in analysis["evidence_contradictions"]]))
    else:
        parts.append("<p class='muted'>No claim was flagged as contradicting the evidence.</p>")
    parts.append("<h2>Contradictions with earlier answers</h2>")
    if analysis["internal_contradictions"]:
        parts.append(_table(["Later claim", "Earlier statement", "Score"], [
            _row([f"<b>{_turn_label(c['later_turn'])}:</b> {escape(c['claim'])}",
                  f"<b>{_turn_label(c['earlier_turn'])}:</b> {escape(c['earlier_statement'])}",
                  f"{c['contradiction']:.2f}"]) for c in analysis["internal_contradictions"]]))
    else:
        parts.append("<p class='muted'>No claim was flagged as contradicting an earlier answer.</p>")
    return "".join(parts)


def _verbal_text(verbal):
    notes = [f"<span class='tag'>{escape(cue)}</span>" for cue in verbal["evasion"]]
    if verbal["hedges"]:
        notes.append("hedges: " + escape(", ".join(verbal["hedges"])))
    if verbal["bolstering"]:
        notes.append("bolstering phrases: " + escape(", ".join(verbal["bolstering"])))
    if verbal["fillers"]:
        notes.append(f"fillers: {verbal['fillers']}")
    details = ", ".join(f"{count} {name}" for name, count in verbal["details"].items() if count)
    notes.append(f"checkable details: {details or 'none'} ({verbal['word_count']} words)")
    return "; ".join(notes)


def _observation_rows(turn):
    deviating = [(o["channel"], o["cue"]) for o in turn["deviations"]]
    rows = []
    for observation in turn["observations"]:
        differs = (observation["channel"], observation["cue"]) in deviating
        rows.append(_row([escape(CHANNELS[observation["channel"]]), escape(observation["cue"]),
                          INTENSITIES[observation.get("intensity", 2)], escape(observation.get("note", "")),
                          "<span class='bad'>differs from baseline</span>" if differs else ""]))
    return rows


def _turn_sections(analysis):
    parts = ["<h2>Turn by turn</h2>"]
    for turn in analysis["turns"]:
        flags = []
        if turn["is_baseline"]:
            flags.append("<span class='tag'>baseline turn</span>")
        if turn["contested"]:
            flags.append("<span class='bad'>contested</span>")
        parts.append(f"<h3>{_turn_label(turn['index'])} {' '.join(flags)}</h3>")
        rows = []
        if turn["question"]:
            rows.append(_row(["Question", f"<span class='question'>{escape(turn['question'])}</span>"]))
        rows.append(_row(["Answer", escape(turn["answer"])]))
        rows.append(_row(["Verbal cues", _verbal_text(turn["verbal"])]))
        flagged = [c for c in turn["claims"] if c["status"] == "contradicted"]
        if flagged:
            rows.append(_row(["Contradicted claims", "<br>".join(
                f"<span class='bad'>{escape(c['claim'])}</span> (vs {escape(c.get('evidence_label', 'evidence'))})"
                for c in flagged)]))
        if "text_emotion" in turn:
            observed = [o["cue"] for o in turn["observations"] if o["channel"] == "emotion"]
            rows.append(_row(["Emotion", f"in the words (model estimate): {escape(turn['text_emotion']['label'])} "
                                         f"({turn['text_emotion']['score']:.2f}); observed by interviewer: "
                                         f"{escape(', '.join(observed)) if observed else 'not recorded'}"]))
        parts.append(_table(None, rows))
        if turn["observations"]:
            parts.append(_table(["Channel", "Observation", "Intensity", "Note", "Baseline"], _observation_rows(turn)))
        else:
            parts.append("<p class='muted'>No behavioral observations recorded for this turn.</p>")
    return "".join(parts)


def _behavior_section(analysis):
    behavior = analysis["behavior"]
    parts = ["<h2>Behavioral observations</h2>",
             "<div class='notice'><b>How to read this section.</b> These are the interviewer's manual notes on "
             "emotion, facial expression, gaze, gestures, posture and voice. Research finds nonverbal cues to be "
             "weak and unreliable signs of deception, and nervous behavior is common in truthful people under "
             "questioning. The report therefore gives no behavioral score. It shows where the notes differ from "
             "the person's own baseline and whether that coincides with contested answers, as a prompt for "
             "follow-up questions only.</div>"]
    if not behavior["total_observations"]:
        return "".join(parts) + "<p class='muted'>No behavioral observations were recorded.</p>"

    if analysis["baseline"]:
        parts.append(f"<h3>Baseline (first {behavior['baseline_turns']} turn(s))</h3>" + _table(
            ["Channel", "Baseline behavior"],
            [_row([escape(CHANNELS[channel]), escape(", ".join(cues))])
             for channel, cues in analysis["baseline"].items()]))
    else:
        parts.append("<p class='muted'>No baseline was recorded, so deviations cannot be computed. Record "
                     "observations during neutral opening questions and set the number of baseline turns.</p>")

    channels = [c for c in CHANNELS if c in behavior["channel_counts"]]
    header = ["Channel"] + [f"T{t['index'] + 1}" for t in analysis["turns"]] + ["Total"]
    rows = []
    for channel in channels:
        counts = [sum(o["channel"] == channel for o in t["observations"]) for t in analysis["turns"]]
        rows.append(_row([escape(CHANNELS[channel])] + [str(c) if c else "" for c in counts] + [str(sum(counts))]))
    rows.append(_row(["<b>Contested answer</b>"]
                     + ["<span class='bad'>yes</span>" if t["contested"] else "" for t in analysis["turns"]] + [""]))
    parts.append("<h3>Observations per turn</h3>" + _table(header, rows))

    parts.append("<h3>Deviations from baseline</h3>" + _table(
        ["Turns after the baseline", "Turns", "Deviating observations", "Per turn"],
        [_row([name.capitalize(), str(d["turns"]), str(d["deviations"]), f"{d['per_turn']:.2f}"])
         for name, d in behavior["deviations"].items()]))
    parts.append("<p class='muted'>A difference between the two rows is a description of this session, not "
                 "evidence: the numbers are small and the observations are subjective.</p>")
    return "".join(parts)


_METHOD = """
<h2>Method and limits</h2>
<ul>
<li><b>Contradictions</b> come from a natural language inference model that compares each sentence of an answer
with the evidence and with earlier answers. Scores close to 1 mean the model is confident, not that it is right.
Read the two sentences yourself before acting on a flag.</li>
<li><b>Verbal cues</b> (evasion, hedges, fillers, amount of detail) are simple pattern counts. They describe the
answer; none of them proves anything on its own.</li>
<li><b>Behavioral observations</b> are typed in by a person and are never scored. See DePaulo et al. (2003),
Psychological Bulletin 129(1); Bond &amp; DePaulo (2006), Personality and Social Psychology Review 10(3); Vrij,
Hartwig &amp; Granhag (2019), Annual Review of Psychology 70.</li>
<li>This report supports an interviewer's own review. It must not be used as a judgment about a real person.</li>
</ul>
"""


def render_report(analysis, generated_at=None):
    session = analysis["session"]
    generated_at = generated_at or datetime.now()
    meta = [f"Subject: {escape(session['subject'] or '')}", f"Session: {escape(session['session_id'] or '')}",
            f"Generated: {generated_at:%Y-%m-%d %H:%M}"]
    if session.get("source"):
        meta.append(f"Source: {escape(session['source'])}")
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>{escape(session['title'] or 'Interrogation report')}</title><style>{_CSS}</style></head><body>"
            f"<h1>{escape(session['title'] or 'Interrogation report')}</h1><p class='meta'>{' | '.join(meta)}</p>"
            "<div class='notice'>This report lists leads for a human reviewer: possible contradictions, verbal "
            "cues and manual behavioral notes. It is not a lie detector and gives no verdict.</div>"
            + _summary_section(analysis) + _evidence_section(analysis) + _contradiction_sections(analysis)
            + _behavior_section(analysis) + _turn_sections(analysis) + _METHOD + "</body></html>")


def save_report(analysis, path, generated_at=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_report(analysis, generated_at))
    return path
