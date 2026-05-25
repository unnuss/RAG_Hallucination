"""
Generate synthetic contradiction training pairs to teach the model
direct factual contradictions (numbers, dates, quantities, yes/no).

For each template we create:
  - a FAITHFUL pair  (context fact == answer fact)        -> label 0 (clean)
  - a CONTRADICTORY pair (context fact != answer fact)     -> label 1 (hallucinated)

These get APPENDED to the existing training_data.jsonl.
We keep the count moderate so synthetic data doesn't swamp the
real RAGTruth examples (33k+), only sharpening contradiction handling.
"""

import json
import random

random.seed(42)

# ---- Building blocks for varied, natural-sounding facts ----
TIME_UNITS = ["days", "weeks", "months", "hours", "years"]
NUMBERS = [3, 5, 7, 10, 14, 15, 21, 24, 30, 45, 60, 90, 100, 180, 365]

# Each template is (context_template, answer_template, kind)
# {a} is the "true" value in context; {b} is the (different) value in the answer.
TEMPLATES = [
    ("Refunds are processed within {a} {unit} of purchase.",
     "Customers can receive a refund within {b} {unit}.", "duration"),
    ("The warranty covers the product for {a} {unit}.",
     "The product is covered under warranty for {b} {unit}.", "duration"),
    ("Shipping typically takes {a} {unit} to arrive.",
     "Delivery usually takes {b} {unit}.", "duration"),
    ("The subscription renews every {a} {unit}.",
     "Your subscription renews every {b} {unit}.", "duration"),
    ("Employees are entitled to {a} {unit} of paid leave per year.",
     "Staff receive {b} {unit} of paid leave annually.", "duration"),
    ("The minimum order value for free shipping is {a} dollars.",
     "Free shipping applies on orders above {b} dollars.", "money"),
    ("The annual membership fee is {a} dollars.",
     "Membership costs {b} dollars per year.", "money"),
    ("The conference will be held in {a}.",
     "The event takes place in {b}.", "place"),
    ("The report must be submitted by {a}.",
     "The deadline for the report is {b}.", "date"),
    ("The maximum file upload size is {a} megabytes.",
     "Files up to {b} megabytes can be uploaded.", "size"),
]

PLACES = ["New York", "London", "Tokyo", "Berlin", "Paris", "Toronto", "Sydney"]
DATES = ["March 15", "June 1", "September 30", "January 10",
         "December 5", "July 22", "April 3"]


def pick_pair(values):
    """Pick two DIFFERENT values from a list."""
    a = random.choice(values)
    b = random.choice([v for v in values if v != a])
    return a, b


def make_examples(n_per_template=40):
    rows = []
    for ctx_t, ans_t, kind in TEMPLATES:
        for _ in range(n_per_template):
            if kind == "duration":
                unit = random.choice(TIME_UNITS)
                a, b = pick_pair(NUMBERS)
                ctx = ctx_t.format(a=a, unit=unit)
                faithful = ans_t.format(b=a, unit=unit)      # same value
                contra = ans_t.format(b=b, unit=unit)         # different value
            elif kind == "money":
                a, b = pick_pair(NUMBERS)
                ctx = ctx_t.format(a=a)
                faithful = ans_t.format(b=a)
                contra = ans_t.format(b=b)
            elif kind == "size":
                a, b = pick_pair(NUMBERS)
                ctx = ctx_t.format(a=a)
                faithful = ans_t.format(b=a)
                contra = ans_t.format(b=b)
            elif kind == "place":
                a, b = pick_pair(PLACES)
                ctx = ctx_t.format(a=a)
                faithful = ans_t.format(b=a)
                contra = ans_t.format(b=b)
            elif kind == "date":
                a, b = pick_pair(DATES)
                ctx = ctx_t.format(a=a)
                faithful = ans_t.format(b=a)
                contra = ans_t.format(b=b)
            else:
                continue

            # Faithful pair -> clean (0)
            rows.append({"context": ctx, "sentence": faithful, "label": 0})
            # Contradictory pair -> hallucinated (1)
            rows.append({"context": ctx, "sentence": contra, "label": 1})
    return rows


new_rows = make_examples(n_per_template=40)
random.shuffle(new_rows)

n_clean = sum(1 for r in new_rows if r["label"] == 0)
n_halluc = sum(1 for r in new_rows if r["label"] == 1)
print(f"Generated {len(new_rows)} synthetic contradiction pairs")
print(f"  Clean (faithful):       {n_clean}")
print(f"  Hallucinated (contra):  {n_halluc}")

# ---- Append to existing training data ----
existing = []
with open("training_data.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            existing.append(json.loads(line))

print(f"\nExisting training pairs: {len(existing)}")

combined = existing + new_rows
random.shuffle(combined)

with open("training_data_v2.jsonl", "w") as f:
    for row in combined:
        f.write(json.dumps(row) + "\n")

total_halluc = sum(1 for r in combined if r["label"] == 1)
print(f"Combined total: {len(combined)} pairs "
      f"({total_halluc} hallucinated, {len(combined)-total_halluc} clean)")
print("\nSaved training_data_v2.jsonl — upload THIS to Colab for retraining.")