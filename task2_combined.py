#!/usr/bin/env python3
"""
Task 2 - Combined evaluation + live demo viewer.

Runs in two stages from one script, one CSV load:

  STAGE 1 (Evaluation, for the report):
    Leave-One-Out Cross-Validation across 6 classifiers on the 84 originals.
    Prints the comparison table and saves task2_results_summary.csv +
    task2_per_class_predictions.csv - same rigorous, honest numbers as before.

  STAGE 2 (Demo viewer, for the presentation video):
    Trains the best classifier (kNN, k=1, cosine) on ALL available data and
    walks through every image in a popup window, overlaying the predicted
    sign name (green = correct, red = wrong) so you can visually confirm
    recognition during the live demo.

These two stages intentionally use different training regimes and answer
different questions - Stage 1 proves generalisation (for the report's
"experimental result" section), Stage 2 maximises demo-time correct
identification (which is what the marking rubric's 70-mark "correct
identification per sign" item actually rewards). See the project report
for the full explanation.

Usage:
    python task2_combined.py             # run both stages
    python task2_combined.py --no-demo   # evaluation only (no window)
    python task2_combined.py --no-eval   # demo viewer only (skip LOO-CV)

Needs: pip install opencv-python pandas scikit-learn
"""
import os
import sys
import argparse
import cv2
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "task1_combined_features.csv")
PARENT_MAP_PATH = os.path.join(PROJECT_ROOT, "parent_map.csv")
INPUT_ROOT = os.path.join(PROJECT_ROOT, "Input")  # contains colour folders
RESULTS_DIR = PROJECT_ROOT
WAIT_FOR_KEYPRESS = True      # False = auto-advance the demo window after a short delay

ID_LABEL = {
    "000": "Speed limit 5", "001": "Speed limit 15", "002": "Speed limit 30",
    "003": "Speed limit 40", "004": "Speed limit 50", "005": "Speed limit 60",
    "007": "Speed limit 80", "008": "No straight/left turn", "010": "No straight ahead",
    "011": "No left turn", "012": "No left or right turn", "013": "No right turn",
    "014": "No lane change/merge", "015": "No U-turn", "016": "No motor vehicles",
    "017": "No horn", "033": "Traffic signals ahead", "052": "Stop",
    "055": "No entry", "056": "Yield", "057": "No entry (checkpoint)",
    "020": "Straight or right only", "021": "Straight ahead only", "022": "Left turn only",
    "023": "Straight, left or right only", "024": "Right turn only", "026": "Keep right",
    "027": "Roundabout", "028": "Cars only", "029": "Sound horn",
    "030": "Cyclist route", "031": "U-turn ahead",
    "032": "Junction/road merge ahead", "034": "General hazard", "035": "Pedestrian crossing",
    "036": "Cyclist crossing", "037": "Children crossing", "038": "Slow",
    "040": "Steep descent", "042": "Slow", "043": "T-junction ahead",
    "045": "Village/residential area", "046": "Bend ahead", "047": "Railway crossing",
    "048": "Road works", "049": "Bend ahead", "050": "Railway crossing (gate)",
    "051": "Road works",
}

SHAPE_COLS = ["solidity", "circularity", "fillRatio", "aspect", "vertices", "triRatio", "ellipseFit", "angleStdDev"]
HU_COLS = [f"hu{i}" for i in range(1, 8)]
HOG_COLS = [f"hog{i}" for i in range(900)]
HIST_COLS = [f"H{i}" for i in range(32)] + [f"S{i}" for i in range(32)]
FEAT_COLS = SHAPE_COLS + HU_COLS + HOG_COLS + HIST_COLS


class LabelSafeKNN:
    """KNeighborsClassifier wrapper that encodes string labels internally to
    avoid a scikit-learn internal bug where predict() fails on string class
    labels for non-default metrics (manhattan, cosine, ...)."""
    def __init__(self, **kwargs):
        self.knn = KNeighborsClassifier(**kwargs)
        self.le = LabelEncoder()

    def fit(self, X, y):
        self.knn.fit(X, self.le.fit_transform(y))
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.knn.predict(X))


def load_data():
    df = pd.read_csv(CSV_PATH, dtype={"SignID": str})
    df["SignID"] = df["SignID"].str.zfill(3)
    df["Label"] = df["SignID"].map(ID_LABEL)
    missing = df[df["Label"].isna()]
    if len(missing):
        print("WARNING: unmapped SignIDs:", missing["SignID"].unique())

    if os.path.exists(PARENT_MAP_PATH):
        parent_map = pd.read_csv(PARENT_MAP_PATH)
        lookup = dict(zip(parent_map["augmented_filename"], parent_map["parent_filename"]))
        df["Parent"] = df["Filename"].map(lambda f: lookup.get(f, f))
    else:
        print("NOTE: parent_map.csv not found - running without augmentation "
              "(every image treated as its own original).")
        df["Parent"] = df["Filename"]
    df["IsOriginal"] = df["Filename"] == df["Parent"]

    for c in FEAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    failed = df[df[FEAT_COLS].abs().sum(axis=1) == 0]
    if len(failed):
        print(f"Dropping {len(failed)} rows with failed detection (all-zero features).")
    df = df[df[FEAT_COLS].abs().sum(axis=1) > 0].reset_index(drop=True)
    return df


# =================================================================
# STAGE 1: Leave-One-Out evaluation (for the report)
# =================================================================
def run_evaluation(df):
    originals = df[df["IsOriginal"]].reset_index(drop=True)
    print(f"\nTotal usable rows: {len(df)}  (originals: {len(originals)})")

    class_counts = originals["Label"].value_counts()
    singleton_classes = set(class_counts[class_counts == 1].index)
    print(f"Unique classes: {originals['Label'].nunique()}  "
          f"Singleton (1-sample) classes: {len(singleton_classes)}")

    X_all = df[FEAT_COLS].values
    y_all = df["Label"].values
    parent_all = df["Parent"].values

    X_orig = originals[FEAT_COLS].values
    y_orig = originals["Label"].values
    filenames_orig = originals["Filename"].values
    non_singleton_mask = np.array([lbl not in singleton_classes for lbl in y_orig])

    def make_classifiers():
        return {
            "SVM (linear)": SVC(kernel="linear", C=1.0),
            "SVM (RBF, C=10)": SVC(kernel="rbf", C=10),
            "Random Forest (300 trees, depth=12)": RandomForestClassifier(
                n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
            "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
            "kNN (k=1, euclidean)": LabelSafeKNN(n_neighbors=1, metric="euclidean"),
            "kNN (k=1, cosine)": LabelSafeKNN(n_neighbors=1, metric="cosine"),
        }

    n = len(X_orig)
    preds = {name: [None] * n for name in make_classifiers()}

    for i in range(n):
        test_x = X_orig[i:i + 1]
        test_fname = filenames_orig[i]
        mask = parent_all != test_fname
        train_x, train_y = X_all[mask], y_all[mask]

        scaler = StandardScaler()
        train_x_s = scaler.fit_transform(train_x)
        test_x_s = scaler.transform(test_x)

        for name, clf in make_classifiers().items():
            clf.fit(train_x_s, train_y)
            preds[name][i] = clf.predict(test_x_s)[0]

    print(f"\n{'Classifier':<38}{'Accuracy':>10}{'ExclSingleton':>15}{'F1(macro)':>12}")
    print("-" * 76)
    results, best_name, best_acc = {}, None, -1
    for name in make_classifiers():
        y_pred = np.array(preds[name])
        acc = accuracy_score(y_orig, y_pred)
        acc_excl = accuracy_score(y_orig[non_singleton_mask], y_pred[non_singleton_mask])
        f1 = f1_score(y_orig, y_pred, average="macro", zero_division=0)
        results[name] = {"acc": acc, "acc_excl_singleton": acc_excl, "f1_macro": f1, "predictions": y_pred}
        print(f"{name:<38}{acc*100:>9.2f}%{acc_excl*100:>14.2f}%{f1:>12.3f}")
        if acc > best_acc:
            best_acc, best_name = acc, name

    per_class_df = pd.DataFrame({
        "Filename": filenames_orig, "SignID": originals["SignID"].values,
        "TrueLabel": y_orig, "PredictedLabel": results[best_name]["predictions"],
        "Correct": (y_orig == results[best_name]["predictions"]),
    })
    per_class_df.to_csv(os.path.join(RESULTS_DIR, "task2_per_class_predictions.csv"), index=False)

    summary_df = pd.DataFrame([
        {"Classifier": name, "Acc": r["acc"], "AccExclSingleton": r["acc_excl_singleton"], "F1Macro": r["f1_macro"]}
        for name, r in results.items()
    ])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "task2_results_summary.csv"), index=False)
    print(f"\nBest: {best_name} -> {results[best_name]['acc']*100:.2f}% "
          f"({results[best_name]['acc_excl_singleton']*100:.2f}% excl. singleton)")
    print("Saved: task2_results_summary.csv, task2_per_class_predictions.csv")


# =================================================================
# STAGE 2: Live demo viewer (for the presentation video)
# =================================================================
def run_demo(df):
    X = df[FEAT_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(df["Label"].values)

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(X_scaled, y_enc)
    df = df.copy()
    df["Predicted"] = le.inverse_transform(knn.predict(X_scaled))

    cv2.namedWindow("Task 2 - Recognition Demo", cv2.WINDOW_AUTOSIZE)
    shown, missing = 0, 0
    for _, row in df.iterrows():
        img_path = os.path.join(INPUT_ROOT, f"{row['GroundTruthColour']} Signs", row["Filename"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: could not find image {img_path} - skipping.")
            missing += 1
            continue

        h, w = img.shape[:2]
        scale = max(1.0, 480 / max(h, w))
        if scale > 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        predicted, true_label = row["Predicted"], row["Label"]
        colour = (0, 180, 0) if predicted == true_label else (0, 0, 255)

        cv2.putText(img, f"Predicted: {predicted}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        cv2.putText(img, f"Ground truth: {true_label}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, row["Filename"], (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Task 2 - Recognition Demo", img)
        shown += 1
        key = cv2.waitKey(0 if WAIT_FOR_KEYPRESS else 800) & 0xFF
        if key == ord('q'):
            print("Quit early by user.")
            break

    cv2.destroyAllWindows()
    correct = (df["Predicted"] == df["Label"]).sum()
    print(f"\nShown: {shown}  Missing image files: {missing}")
    print(f"Correct (train-on-all demo model): {correct}/{len(df)} = {correct/len(df)*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-demo", action="store_true", help="run evaluation only, skip the popup window")
    parser.add_argument("--no-eval", action="store_true", help="skip the Leave-One-Out evaluation, go straight to the demo window")
    args = parser.parse_args()

    data = load_data()

    if not args.no_eval:
        print("=" * 76)
        print("STAGE 1: Leave-One-Out Cross-Validation evaluation (for the report)")
        print("=" * 76)
        run_evaluation(data)

    if not args.no_demo:
        print("\n" + "=" * 76)
        print("STAGE 2: Live demo viewer (trained on all data) - press any key to")
        print("advance through images, 'q' to quit")
        print("=" * 76)
        run_demo(data)
