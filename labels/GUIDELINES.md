# Labeling guidelines

Each row of `labeling_template.csv` is one suspect–witness pair. Decide whether the **witness statement is
consistent with the suspect's account**, using only the two texts.

## Labels

| Label | Use it when |
|---|---|
| `consistent` | The witness agrees with the suspect on the central facts: what happened, who did what, when and where. Extra details are fine as long as they do not conflict. |
| `partially_consistent` | The witness agrees on some central facts but conflicts on a peripheral detail, or is uncertain or did not see the decisive moment ("I couldn't see whether she opened the gate"). |
| `contradictory` | The witness conflicts with at least one **central** claim of the suspect. Example: suspect "I never left the house that evening", witness "she was at the bus stop at 9 PM". |

A detail that only one side mentions is **not** a contradiction by itself. It becomes one only when the two
accounts cannot both be true.

## Rules

1. Label independently. Do not discuss pairs with the other annotator and do not look at any model output
   (the GUI, `results/`, `emotion_outputs/`) before you finish.
2. Judge consistency, not truthfulness: the question is never who is lying.
3. If two labels seem equally right, pick the one you find more defensible and explain why in `notes`.
4. Fill `label` with exactly one of `consistent`, `partially_consistent`, `contradictory`.

## Process

1. Copy the template to your own file: `annotator_A.csv`, `annotator_B.csv`, and so on. Never edit the template itself.
2. Two annotators label all pairs independently.
3. Measure agreement (Cohen's kappa), then discuss the disagreements and record the agreed label in `gold.csv`.
   Only `gold.csv` is used to evaluate the system.

## Opening the CSV in Excel

The file is comma-separated UTF-8. With a Turkish regional setting, Excel expects `;` and may put everything
into one column when you double-click it. Use **Data → From Text/CSV**, choose UTF-8 and the comma delimiter,
or open the file in Google Sheets or LibreOffice. Save it as CSV (UTF-8) again.
