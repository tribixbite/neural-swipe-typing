#!/usr/bin/env python3
"""Quick analysis of the filtered dataset."""

import json
from collections import Counter

# Read first 1000 lines and analyze
word_counter = Counter()
canvas_sizes = []
total = 0

with open('data/futo/train_filtered.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        data = json.loads(line)
        word_counter[data['word']] += 1
        canvas_sizes.append((data['canvas_width'], data['canvas_height']))
        total += 1

print(f"Analyzed {total} samples from filtered dataset")
print(f"\nTop 20 most common words:")
for word, count in word_counter.most_common(20):
    print(f"  {word:15} {count:4}")

print(f"\nSample canvas dimensions (first 10):")
for i, (w, h) in enumerate(canvas_sizes[:10]):
    print(f"  {w:6.1f} x {h:6.1f}")