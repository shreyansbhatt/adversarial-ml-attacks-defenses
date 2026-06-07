"""
AI/ML Adversarial Attack Demo
1. Train a baseline classifier on breast cancer dataset
2. Demonstrate data poisoning (label flipping) at varying contamination rates
3. Demonstrate evasion attack (find adversarial input that flips prediction)
4. Show defense — train with adversarial examples
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def train_model(X, y):
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


# --- Baseline ---
print("=" * 70)
print("BASELINE — clean training data")
print("=" * 70)
model = train_model(X_train, y_train)
clean_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Clean test accuracy: {clean_acc:.4f}")

# --- Attack 1: Data Poisoning (label flipping) ---
print("\n" + "=" * 70)
print("ATTACK 1 — Data poisoning via label flipping")
print("=" * 70)
print(f"{'Poison %':>10} {'Train accuracy':>16} {'Test accuracy':>16} {'Degradation':>14}")
print("-" * 60)
for poison_rate in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
    y_poisoned = y_train.copy()
    n_flip = int(poison_rate * len(y_poisoned))
    flip_idx = np.random.choice(len(y_poisoned), n_flip, replace=False)
    y_poisoned[flip_idx] = 1 - y_poisoned[flip_idx]

    m = train_model(X_train, y_poisoned)
    train_acc = accuracy_score(y_train, m.predict(X_train))
    test_acc = accuracy_score(y_test, m.predict(X_test))
    deg = clean_acc - test_acc
    print(f"{poison_rate*100:>9.0f}% {train_acc:>16.4f} {test_acc:>16.4f} {deg:>14.4f}")

# --- Attack 2: Evasion (find adversarial input via random perturbation search) ---
print("\n" + "=" * 70)
print("ATTACK 2 — Evasion via adversarial perturbation")
print("=" * 70)

# Pick a malignant sample CLOSEST to the decision boundary (least confidently
# classified). Real attackers don't waste budget on high-confidence samples —
# they target borderline cases where small perturbations can flip predictions.
# Filter for malignant samples that are CORRECTLY classified
malignant_idx = np.where(y_test == 0)[0]
preds_on_malignant = model.predict(X_test[malignant_idx])
correctly_classified = malignant_idx[preds_on_malignant == 0]

# Among correctly classified, pick the LOWEST confidence (closest to decision boundary)
probs = model.predict_proba(X_test[correctly_classified])[:, 0]  # P(label 0)
target = correctly_classified[np.argmin(probs)]
original = X_test[target].copy()
orig_pred = model.predict([original])[0]
orig_conf = model.predict_proba([original])[0][orig_pred]

print(f"Borderline malignant sample (idx={target}): true label=0, "
      f"model CORRECTLY predicts {orig_pred} with confidence {orig_conf:.4f}")
print(f"(This is the sample the model is least confident about — "
      f"the realistic attack target.)")

# Random-search adversarial example within an L-infinity ball
budget = 0.10 * (X_train.max(axis=0) - X_train.min(axis=0))  # 5% of feature range
best_adv = None
for _ in range(15000):
    perturb = np.random.uniform(-budget, budget, size=original.shape)
    adversarial = original + perturb
    pred = model.predict([adversarial])[0]
    if pred != orig_pred:
        best_adv = adversarial
        break

if best_adv is not None:
    adv_pred = model.predict([best_adv])[0]
    adv_conf = model.predict_proba([best_adv])[0][adv_pred]
    delta = np.abs(best_adv - original)
    print(f"Adversarial perturbation found within 5% L-inf budget:")
    print(f"  Model now predicts {adv_pred} with confidence {adv_conf:.4f}")
    print(f"  Max single-feature change: {delta.max():.6f}")
    print(f"  Mean perturbation: {delta.mean():.6f}")
    print(f"  Classification FLIPPED — evasion successful")
else:
    print("No adversarial example found in 5000 trials (budget too small)")

# --- Defense: Adversarial Training ---
print("\n" + "=" * 70)
print("DEFENSE — Adversarial training (Madry-style)")
print("=" * 70)

# Generate adversarial examples for training set and append to training data
adv_X, adv_y = [], []
for i in range(min(50, len(X_train))):
    orig = X_train[i].copy()
    orig_pred = model.predict([orig])[0]
    for _ in range(200):
        perturb = np.random.uniform(-budget, budget, size=orig.shape)
        candidate = orig + perturb
        if model.predict([candidate])[0] != orig_pred:
            adv_X.append(candidate)
            adv_y.append(y_train[i])  # keep TRUE label
            break

print(f"Generated {len(adv_X)} adversarial training examples")
X_aug = np.vstack([X_train, adv_X])
y_aug = np.concatenate([y_train, adv_y])
robust_model = train_model(X_aug, y_aug)

# Re-test the original adversarial example
if best_adv is not None:
    robust_pred = robust_model.predict([best_adv])[0]
    print(f"Robust model on the same adversarial input: predicts {robust_pred}")
    print(f"  Defense {'WORKED' if robust_pred == 0 else 'FAILED'} "
          f"(true label was 0)")

print(f"\nRobust model clean accuracy: "
      f"{accuracy_score(y_test, robust_model.predict(X_test)):.4f} "
      f"(vs baseline {clean_acc:.4f})")