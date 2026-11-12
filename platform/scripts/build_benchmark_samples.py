"""Build the two benchmark samples shipped in ``agentlens_platform/examples/data``.

Deterministic (fixed seed), so the committed files can be regenerated and
checked byte for byte.

    python scripts/build_benchmark_samples.py --squad dev-v1.1.json \\
        --scientsbank test-ud-00001.parquet

Sources:

* SQuAD v1.1 dev set (Rajpurkar et al., 2016), CC BY-SA 4.0.
  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
* SciEntsBank, SemEval-2013 Task 7 (Dzikovska et al., 2013), CC BY 4.0,
  "unseen domains" test split as published at
  https://huggingface.co/datasets/nkazi/SciEntsBank
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "agentlens_platform" / "examples" / "data"

SEED = 20261112
QUESTIONS = 100
DISTRACTOR_PARAGRAPHS = 300
GRADED_ANSWERS = 150

#: Label ids as published. The 2-way scheme defined by the dataset authors
#: merges everything except "correct" into "incorrect".
SCIENTSBANK_LABELS = [
    "correct",
    "contradictory",
    "partially_correct_incomplete",
    "irrelevant",
    "non_domain",
]


def build_squad(path: Path) -> dict:
    rng = random.Random(SEED)
    articles = json.loads(path.read_text(encoding="utf-8"))["data"]

    paragraphs = []
    for article in articles:
        for index, paragraph in enumerate(article["paragraphs"]):
            paragraphs.append(
                {
                    "id": f"{article['title']}#{index}",
                    "title": article["title"].replace("_", " "),
                    "text": paragraph["context"],
                    "qas": paragraph["qas"],
                }
            )
    rng.shuffle(paragraphs)

    # One question per paragraph, so no two questions share their evidence and
    # retrieval has to find a different passage each time.
    chosen, corpus = [], []
    for paragraph in paragraphs:
        if len(chosen) < QUESTIONS and 400 <= len(paragraph["text"]) <= 1200:
            qa = rng.choice(paragraph["qas"])
            answers = sorted({a["text"].strip() for a in qa["answers"] if a["text"].strip()})
            if not answers:
                continue
            chosen.append(
                {
                    "id": qa["id"],
                    "question": qa["question"].strip(),
                    "answers": answers,
                    "paragraph_id": paragraph["id"],
                }
            )
            corpus.append({k: paragraph[k] for k in ("id", "title", "text")})
        elif len(corpus) < QUESTIONS + DISTRACTOR_PARAGRAPHS and len(paragraph["text"]) <= 1200:
            corpus.append({k: paragraph[k] for k in ("id", "title", "text")})
    corpus.sort(key=lambda p: p["id"])
    assert len(chosen) == QUESTIONS, len(chosen)
    return {
        "source": "SQuAD v1.1 dev set (Rajpurkar et al., 2016)",
        "license": "CC BY-SA 4.0",
        "url": "https://rajpurkar.github.io/SQuAD-explorer/",
        "seed": SEED,
        "questions": chosen,
        "corpus": corpus,
    }


def build_scientsbank(path: Path) -> dict:
    import pandas as pd

    rng = random.Random(SEED)
    frame = pd.read_parquet(path)
    rows = frame.to_dict("records")
    # Keep prompts a reasonable length for a rate-limited judge.
    rows = [
        r
        for r in rows
        if len(r["question"]) <= 600 and 15 <= len(r["student_answer"]) <= 400
    ]
    rows.sort(key=lambda r: r["id"])
    sample = rng.sample(rows, GRADED_ANSWERS)
    sample.sort(key=lambda r: r["id"])
    items = [
        {
            "id": r["id"],
            "question": r["question"].strip(),
            "reference_answer": r["reference_answer"].strip(),
            "student_answer": r["student_answer"].strip(),
            "label_5way": SCIENTSBANK_LABELS[int(r["label"])],
            "label_2way": "correct" if int(r["label"]) == 0 else "incorrect",
        }
        for r in sample
    ]
    return {
        "source": "SciEntsBank, SemEval-2013 Task 7 (Dzikovska et al., 2013), unseen-domains test split",
        "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/nkazi/SciEntsBank",
        "seed": SEED,
        "labelling": "Human expert annotators graded each student answer against the reference "
        "answer. label_2way is the 2-way scheme defined by the dataset authors.",
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad", type=Path, required=True)
    parser.add_argument("--scientsbank", type=Path, required=True)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    squad = build_squad(args.squad)
    (OUT / "squad_sample.json").write_text(
        json.dumps(squad, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    seb = build_scientsbank(args.scientsbank)
    (OUT / "scientsbank_sample.json").write_text(
        json.dumps(seb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for item in seb["items"]:
        counts[item["label_5way"]] = counts.get(item["label_5way"], 0) + 1
    print(f"squad: {len(squad['questions'])} questions, {len(squad['corpus'])} paragraphs")
    print(f"scientsbank: {len(seb['items'])} graded answers, {counts}")


if __name__ == "__main__":
    main()
