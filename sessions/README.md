# Sample sessions

The `alibi-*.json` files are short interrogation dialogues converted from
[solarkyle/alibi-interrogation-dataset](https://huggingface.co/datasets/solarkyle/alibi-interrogation-dataset)
(Apache-2.0). They are **synthetic**: an LLM wrote both the detective and the suspect from a character card.

How they were converted (`python external_datasets.py samples`):

- Each detective question and suspect reply is one turn.
- Stage directions in the reply, such as `*picks at fingernails*`, were removed from the answer text and stored
  as behavioral observations, with the channel guessed from keywords.
- The "Secret Knowledge" lines of the character card became the evidence ("Fact 1", "Fact 2", ...).
- `ground_truth.alibi` records whether the card calls the alibi a lie or the truth. The analysis never reads it;
  it exists for `benchmarks.py alibi`.

Sessions you save from the app are stored in this folder as well.
