#!/usr/bin/env python3
"""Convert a pinned BANKING77 snapshot to local-classifier and Jev-compatible files."""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPOSITORY = "https://github.com/PolyAI-LDN/task-specific-datasets"
REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
CHECKSUMS = {
    "banking_data/train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "banking_data/test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "banking_data/categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
    "README.md": "7b9ce9069931cc2de89dce8c7ffb53d07806cf413601ccf133b9af5227b3a373",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fetch_source(directory, offline=False):
    for name, expected in CHECKSUMS.items():
        path = directory / name
        if not path.exists():
            if offline:
                raise ValueError(f"Missing cached source file: {path}")
            url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if sha256(data) != expected:
                raise ValueError(f"Upstream checksum mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if sha256(path.read_bytes()) != expected:
            raise ValueError(f"Cached source checksum mismatch: {path}")


def load_csv(path, split, labels):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for i, row in enumerate(rows):
        if not row.get("text", "").strip() or row.get("category") not in labels:
            raise ValueError(f"Invalid source row {i + 1} in {path}")
        result.append({"id": f"banking77-{split}-{i:05d}", "state": row["text"],
                       "label": row["category"], "group": sha256(text_key(row["text"]).encode())})
    return result


def clean_splits(train, test):
    """Keep test examples first; remove duplicate training exposure without reading test labels."""
    audit = []

    def unique(rows, split):
        kept = {}
        for row in rows:
            key = text_key(row["state"])
            if key in kept:
                if kept[key]["label"] != row["label"]:
                    raise ValueError(f"Conflicting labels for normalized duplicate {row['id']}")
                audit.append({"id": row["id"], "reason": f"duplicate_within_{split}", "keptId": kept[key]["id"]})
            else:
                kept[key] = row
        return list(kept.values())

    clean_test = unique(test, "test")
    test_keys = {text_key(row["state"]) for row in clean_test}
    clean_train = []
    for row in unique(train, "train"):
        if text_key(row["state"]) in test_keys:
            audit.append({"id": row["id"], "reason": "input_already_in_official_test"})
        else:
            clean_train.append(row)
    return clean_train, clean_test, audit


def validation_split(rows, fraction=0.2, seed=42):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    train, validation = [], []
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda row: sha256(f"{seed}:{row['group']}".encode()))
        if len(ordered) < 2:
            raise ValueError(f"Not enough independent examples for {label}")
        count = max(1, min(len(ordered) - 1, int(len(ordered) * fraction)))
        validation.extend(ordered[:count])
        train.extend(ordered[count:])
    return train, validation


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(source, destination, seed=42):
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    labels = json.loads((source / "banking_data/categories.json").read_text())
    original_train = load_csv(source / "banking_data/train.csv", "train", labels)
    original_test = load_csv(source / "banking_data/test.csv", "test", labels)
    cleaned_train, test, audit = clean_splits(original_train, original_test)
    train, validation = validation_split(cleaned_train, seed=seed)
    task = {"questions": {"intent": {
        "type": "choice",
        "instructions": "Classify the primary intent of this English online-banking customer query. Choose exactly one of the supplied BANKING77 intent labels.",
        "criteria": {label: label.replace("_", " ").rstrip("?") for label in labels},
    }}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        write_json(staging / "task.json", task)
        for split, rows in [("train", train), ("validation", validation), ("test", test)]:
            # The canonical benchmark uses known human labels. Teacher inputs deliberately omit them.
            write_jsonl(staging / f"{split}.jsonl", rows)
            write_jsonl(staging / f"{split}.inputs.jsonl", ({k: row[k] for k in ("id", "state", "group")} for row in rows))
            write_jsonl(staging / f"{split}.outputs.jsonl", ({
                "id": row["id"], "answers": {"intent": {"type": "choice", "choice": row["label"]}},
                "source": {"dataset": "BANKING77", "annotation": "human", "revision": REVISION},
            } for row in rows))
        write_json(staging / "excluded.json", audit)
        shutil.copyfile(source / "LICENSE", staging / "LICENSE.txt")
        manifest = {
            "dataset": "BANKING77", "repository": REPOSITORY, "revision": REVISION,
            "license": "CC-BY-4.0", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Casanueva, Temcinas, Gerz, Henderson, and Vulic. Efficient Intent Detection with Dual Sentence Encoders. NLP4ConvAI, ACL 2020.",
            "paper": "https://arxiv.org/abs/2003.04807", "sourceChecksums": CHECKSUMS,
            "labels": len(labels), "seed": seed, "originalCounts": {"train": len(original_train), "test": len(original_test)},
            "preparedCounts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "exclusions": dict(Counter(row["reason"] for row in audit)),
            "splitPolicy": "Preserve official test membership except duplicate normalized inputs; remove training overlap; stratified 20% validation from remaining official train.",
            "normalizationForDeduplication": "NFKC, casefold, collapse whitespace. Original text is retained for model input.",
            "targetPolicy": "Original human labels. Hard Choice outputs are a format conversion, not Jev responses or calibrated probability distributions.",
            "criteriaPolicy": "Descriptions mechanically expand label names; they are not official definitions and should be reviewed before teacher labelling.",
            "perClass": {label: {name: sum(row["label"] == label for row in rows) for name, rows in [("train", train), ("validation", validation), ("test", test)]} for label in sorted(labels)},
        }
        manifest["preparedChecksums"] = {path.name: sha256(path.read_bytes()) for path in sorted(staging.iterdir())}
        write_json(staging / "manifest.json", manifest)
        (staging / "README.md").write_text(
            "# BANKING77, converted to Jev-compatible records\n\n"
            "Source: PolyAI / Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders (2020).\n"
            f"{REPOSITORY}, revision `{REVISION}`.\n\n"
            "Dataset licence: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); full licence in LICENSE.txt.\n\n"
            "Changes: conversion to JSONL, normalized duplicate removal, removal of train/test input overlap, "
            "and a deterministic validation split. See manifest.json and excluded.json for exact details.\n\n"
            "train.jsonl / validation.jsonl / test.jsonl contain human reference labels. "
            "The *.inputs.jsonl files omit labels for teacher requests. The *.outputs.jsonl files encode "
            "the existing human labels as hard Choice answers, not model-generated answers. "
            "Do not invent soft probabilities from these labels.\n\n"
            "When training on a teacher's answers, use input-only files and the separate teacher responses. "
            "Using the gold-labelled train.jsonl with teacher outputs would cause the SDK's importer to prefer the human labels. "
            "Keep test.jsonl out of training and model selection.\n", encoding="utf-8")
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/banking77-source"))
    parser.add_argument("--out", type=Path, default=Path("datasets/banking77"))
    parser.add_argument("--offline", action="store_true", help="Require already-downloaded source files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    fetch_source(args.source, args.offline)
    manifest = prepare(args.source, args.out, args.seed)
    print(json.dumps({k: manifest[k] for k in ("dataset", "revision", "labels", "preparedCounts", "exclusions")}, indent=2))


if __name__ == "__main__":
    main()
