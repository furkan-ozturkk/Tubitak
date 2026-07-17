#!/usr/bin/env python3
"""
build_pilot.py

Section 3.1 orchestrator: runs the three tiers (deterministic / semantic /
hard) and merges their output into a single pilot dataset. Every tunable
value (concurrency limit, target size, difficulty mix) is read from
scale_config.yaml so this script needs no code changes across Phase 1/2/3
(Section 3.2) -- only the config file changes.

Usage:
  python3 build_pilot.py --config /app/scale_config.yaml --corpus-dir /data/loghub \
      --out /output/pilot/questions.json --review-dir /output/pilot/review/groundedness
"""
import argparse
import json
from pathlib import Path

import yaml

from layer1_deterministic import generate_all as generate_layer1
from layer2_semantic import generate_all as generate_layer2
from layer3_hard import generate_all as generate_layer3
from ollama_client import OllamaClient


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review-dir", type=Path, required=True)
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    max_parallel = config.get("concurrency", {}).get("max_parallel_model_calls", 4)
    target_total = config.get("run", {}).get("target_total_questions", 100)
    difficulty_mix = config.get("difficulty_mix", {"easy": 0.70, "medium": 0.20, "hard": 0.10})

    print("=== Layer 1 (easy / deterministic) ===")
    layer1_records = generate_layer1(args.corpus_dir)

    client = OllamaClient(max_parallel_calls=max_parallel)

    print("\n=== Layer 2 (medium / semantic) ===")
    layer2_records = generate_layer2(args.corpus_dir, client)

    print("\n=== Layer 3 (hard) ===")
    layer3_records = generate_layer3(args.corpus_dir, client, args.review_dir)

    all_records = layer1_records + layer2_records + layer3_records
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(all_records)
    print("\n=== SUMMARY ===")
    print(f"easy={len(layer1_records)} medium={len(layer2_records)} hard={len(layer3_records)} "
          f"total={total} (target ~{target_total}, configured mix={difficulty_mix})")
    print(f"Wrote pilot dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
