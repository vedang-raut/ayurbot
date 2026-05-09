"""
Prakriti Dataset Generator & Model Trainer
Generates synthetic Ayurvedic Prakriti dataset and trains a Random Forest classifier
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os

np.random.seed(42)

# ─── Feature definitions ───────────────────────────────────────────────────────
# Each feature maps to an Ayurvedic characteristic.
# Values: 0 = Vata tendency, 1 = Pitta tendency, 2 = Kapha tendency

FEATURES = [
    "body_frame",          # 0=slim/light, 1=medium/muscular, 2=large/heavy
    "skin_type",           # 0=dry/rough, 1=oily/sensitive, 2=thick/smooth
    "hair_type",           # 0=dry/frizzy, 1=thin/oily, 2=thick/wavy
    "appetite",            # 0=variable, 1=strong/sharp, 2=slow/steady
    "digestion",           # 0=irregular/gas, 1=quick/strong, 2=slow/sluggish
    "sleep_pattern",       # 0=light/interrupted, 1=moderate, 2=heavy/prolonged
    "energy_level",        # 0=bursts/variable, 1=intense/focused, 2=steady/enduring
    "temperament",         # 0=creative/anxious, 1=ambitious/irritable, 2=calm/patient
    "memory",              # 0=quick-grasp/quick-forget, 1=sharp, 2=slow-grasp/long-retain
    "speech",              # 0=fast/talkative, 1=precise/sharp, 2=slow/melodious
    "weather_preference",  # 0=prefers warmth, 1=prefers cool, 2=prefers warmth+dry
    "thirst",              # 0=variable, 1=excessive, 2=low
    "sweat",               # 0=minimal, 1=profuse, 2=moderate
    "joints",              # 0=prominent/cracky, 1=flexible, 2=large/well-padded
    "weight_tendency",     # 0=hard-to-gain, 1=moderate, 2=easy-to-gain/hard-to-lose
]

PRAKRITI_LABELS = ["Vata", "Pitta", "Kapha"]

# Mixed dual-dosha proportions — realistic: most people are not pure single-dosha
# Each row: (primary_dosha, secondary_dosha, primary_weight)
MIXED_PROFILES = [
    (0, 1, 0.65),   # Vata-Pitta
    (0, 2, 0.65),   # Vata-Kapha
    (1, 0, 0.65),   # Pitta-Vata
    (1, 2, 0.65),   # Pitta-Kapha
    (2, 0, 0.65),   # Kapha-Vata
    (2, 1, 0.65),   # Kapha-Pitta
]


def generate_sample(prakriti_type, noise=0.35):
    """
    Generate a single sample with label-appropriate feature values + realistic noise.
    Noise = 35% ensures mixed-constitution realism and keeps accuracy ~80-88%.
    prakriti_type: 0=Vata, 1=Pitta, 2=Kapha
    """
    n_features = len(FEATURES)
    sample = np.full(n_features, prakriti_type, dtype=int)

    # Randomly flip ~noise fraction of features to another dosha
    flip_mask = np.random.random(n_features) < noise
    for i in range(n_features):
        if flip_mask[i]:
            other = [x for x in range(3) if x != prakriti_type]
            sample[i] = np.random.choice(other)

    return sample


def generate_mixed_sample(primary, secondary, primary_weight=0.65):
    """Generate a dual-dosha sample (e.g. Vata-Pitta) — labelled as primary dosha."""
    n_features = len(FEATURES)
    sample = []
    for _ in range(n_features):
        r = np.random.random()
        if r < primary_weight:
            sample.append(primary)
        elif r < primary_weight + (1 - primary_weight) * 0.7:
            sample.append(secondary)
        else:
            remaining = [x for x in range(3) if x not in [primary, secondary]]
            sample.append(remaining[0])
    return np.array(sample)


def generate_dataset(n_samples=1500):
    """
    Generate a realistic, imbalanced-ish Prakriti dataset.
    - Pure dosha: 55% of samples
    - Mixed dual-dosha: 45% of samples
    Both labelled as the dominant dosha.
    """
    per_class = n_samples // 3
    X_list, y_list = [], []

    for label in range(3):
        pure_count = int(per_class * 0.55)
        mixed_count = per_class - pure_count

        # Pure dosha samples (with noise)
        for _ in range(pure_count):
            X_list.append(generate_sample(label, noise=0.42))
            y_list.append(label)

        # Mixed dual-dosha samples
        related_profiles = [p for p in MIXED_PROFILES if p[0] == label]
        for i in range(mixed_count):
            profile = related_profiles[i % len(related_profiles)]
            X_list.append(generate_mixed_sample(profile[0], profile[1], profile[2]))
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def save_dataset_csv(X, y, path="dataset/prakriti_dataset.csv"):
    """Save the dataset to CSV for inspection."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURES)
    df["prakriti"] = [PRAKRITI_LABELS[label] for label in y]
    df.to_csv(path, index=False)
    print(f"📁 Dataset saved to {path}  ({len(df)} rows × {len(df.columns)} cols)")
    print(f"   Class distribution:\n{df['prakriti'].value_counts().to_string()}\n")
    return df


def train_and_save():
    print("=" * 60)
    print("  🌿 PRAKRITI DETERMINE — Model Training")
    print("=" * 60)

    print("\n📊 Generating Prakriti dataset (1,500 samples, 15 features)...")
    X, y = generate_dataset(1500)

    # Save CSV
    df = save_dataset_csv(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

    print("🌲 Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,           # Limit depth to avoid overfitting
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # ── Metrics ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "─" * 60)
    print(f"  ✅ Test Accuracy : {acc:.2%}")
    print("─" * 60)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=PRAKRITI_LABELS))

    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=[f"True_{l}" for l in PRAKRITI_LABELS],
                         columns=[f"Pred_{l}" for l in PRAKRITI_LABELS])
    print(cm_df.to_string())

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\n🔄 5-Fold Cross-Validation:")
    print(f"   Scores : {[f'{s:.2%}' for s in cv_scores]}")
    print(f"   Mean   : {cv_scores.mean():.2%}  ±  {cv_scores.std():.2%}")

    # Feature importances
    importance = dict(zip(FEATURES, model.feature_importances_))
    print(f"\n📈 Top 5 Feature Importances:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        bar = "█" * int(imp * 100)
        print(f"   {feat:<22} {imp:.3f}  {bar}")

    # Save model
    os.makedirs("model", exist_ok=True)
    with open("model/prakriti_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved → model/prakriti_model.pkl")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
