# fix_labels.py
import os

LABELS = "char_dataset/labels.txt"

def sort_labels():
    items = []
    for line in open(LABELS, "r", encoding="utf-8"):
        if "\t" not in line: continue
        fname, ch = line.strip().split("\t")
        items.append((fname, ch))

    items.sort(key=lambda x: (x[1], x[0]))

    with open(LABELS, "w", encoding="utf-8") as f:
        for fname, ch in items:
            f.write(f"{fname}\t{ch}\n")

    print("✔ labels.txt sorted")

if __name__ == "__main__":
    sort_labels()
