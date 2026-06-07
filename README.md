# Adversarial Machine Learning — Poisoning, Evasion & Robust Defense

A single-file, runnable demonstration of three core adversarial-ML phenomena
against a real classifier, plus a working defense. Train a model, poison it,
evade it, then harden it — all in one script you can read top to bottom.

> **For education and authorized security research only.** Every technique here
> runs against a local model on a public dataset. Do not apply these methods to
> systems you do not own or have explicit permission to test.

---

## What it demonstrates

The script (`adversarial_demo.py`) walks through the adversarial-ML lifecycle on
the scikit-learn breast-cancer dataset with a `GradientBoostingClassifier`:

| Stage | Technique | What you see |
|-------|-----------|--------------|
| **Baseline** | Clean training | Reference test accuracy |
| **Attack 1 — Data Poisoning** | Label flipping at 0–40% contamination | Accuracy degradation curve as poison rate rises |
| **Attack 2 — Evasion** | L∞-bounded perturbation search against a borderline sample | A correctly-classified malignant case flipped to benign within a 5% feature budget |
| **Defense** | Adversarial training (Madry-style) | The same adversarial input correctly re-classified after augmentation |

The evasion attack deliberately targets the sample **closest to the decision
boundary** (lowest model confidence) — the realistic attacker behavior, rather
than wasting perturbation budget on high-confidence examples.

### Framework mapping

- **MITRE ATLAS** — `AML.T0020` (Poison Training Data), `AML.T0043` (Craft
  Adversarial Data / Evasion), `AML.T0042`-style adversarial-training mitigation.
- **OWASP ML Top 10** — ML01 (Input Manipulation / Evasion), ML02 (Data
  Poisoning).

---

## Run it

```bash
pip install -r requirements.txt
python adversarial_demo.py
```

Runs in seconds on CPU. Output is printed as labeled tables and verdicts for
each stage.

---

## Why it matters

Poisoning and evasion are the two attack families that every deployed ML model
inherits for free. This POC shows both with measurable effect and then shows
that **adversarial training meaningfully restores robustness** — while also
making the cost explicit (clean accuracy vs. robust accuracy trade-off).

---

## Skills demonstrated

Adversarial Machine Learning · Data Poisoning · Evasion Attacks · Adversarial
Training · AI Red Teaming · ML Security · MITRE ATLAS · scikit-learn · Python

---

## License

MIT — see [LICENSE](LICENSE).
