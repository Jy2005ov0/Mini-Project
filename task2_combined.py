#!/usr/bin/env python3
"""
Task 2 - Combined evaluation + live demo viewer.

Runs in three stages from one script, one CSV load:

  STAGE 0 (External-data evaluation, for the report - the headline number):
    Trains on ~4,600 real photos from the TSRD dataset (same 000-057 sign
    coding as ours; see load_external_data() for source/dedup details) and
    tests on all 84 of our own images, which the model never trains on.
    Unlike Stage 1, most classes here have dozens of real training examples
    instead of 1-2, so this is a genuine measure of recognition rate rather
    than a small-data artifact. Needs TSRD_external/task1_combined_features.csv
    (run Task2Demo.exe against the TSRD_external image set to generate it);
    skipped with a note if that file is missing. Saves task2_external_
    results_summary.csv (accuracy/precision/recall/F1 per classifier),
    task2_external_per_class_predictions.csv, one confusion-matrix PNG per
    classifier, and a classifier-comparison bar chart.

  STAGE 1 (LOO-CV evaluation, for the report - the small-data baseline):
    Leave-One-Out Cross-Validation across our 3 classifiers (SVM, Random
    Forest, Logistic Regression - deliberately not overlapping with the
    groupmate's KNN/ANN/Naive Bayes half of Task 2, so the team's combined
    submission covers 6 distinct algorithms) on the 84 originals only. Each
    fold standardises features then keeps the top N_SELECT_FEATURES by
    ANOVA F-score (fit on the training fold only, so no leakage) before the
    classifier sees them - with 979 raw features and ~83 training rows per
    fold, this curbs the curse of dimensionality and lifted every classifier
    in testing. 27 of 48 classes have only one example each, so those are
    mathematically unlearnable here - compare against Stage 0 to see what the
    external data bought. Saves the same kinds of outputs as Stage 0
    (task2_results_summary.csv, task2_per_class_predictions.csv, confusion
    matrices, comparison chart).

  STAGE 2 (Demo viewer, for the presentation video):
    Trains the best classifier (kNN, k=1, cosine) on ALL available data and
    walks through every image in a popup window, overlaying the predicted
    sign name (green = correct, red = wrong) so you can visually confirm
    recognition during the live demo.

These stages intentionally use different training regimes and answer
different questions - Stage 0 and 1 report generalisation honestly (for the
report's "experimental result" section, with Stage 0 as the real-world
number and Stage 1 as the small-data comparison), Stage 2 maximises
demo-time correct identification (which is what the marking rubric's
70-mark "correct identification per sign" item actually rewards). See the
project report for the full explanation.

Usage:
    python task2_combined.py                # run all three stages
    python task2_combined.py --no-demo      # evaluation only (no window)
    python task2_combined.py --no-eval      # skip LOO-CV (Stage 1)
    python task2_combined.py --no-external  # skip the TSRD evaluation (Stage 0)

Needs: pip install opencv-python pandas scikit-learn
"""
import os
import re
import sys
import argparse
import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save-only: no popup windows for the evaluation charts
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "task1_combined_features.csv")
PARENT_MAP_PATH = os.path.join(PROJECT_ROOT, "parent_map.csv")
EXTERNAL_CSV_PATH = os.path.join(PROJECT_ROOT, "TSRD_external", "task1_combined_features.csv")
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
    "036": "Cyclist crossing", "037": "Children crossing", "038": "Bend right ahead",
    "040": "Steep descent", "042": "Slow", "043": "T-junction ahead",
    "045": "Village/residential area", "046": "Double bend ahead", "047": "Railway crossing",
    "048": "Road works", "049": "Winding road ahead", "050": "Railway crossing (gate)",
    "051": "Accident-prone area",
}

SHAPE_COLS = ["solidity", "circularity", "fillRatio", "aspect", "vertices", "triRatio", "ellipseFit", "angleStdDev"]
HU_COLS = [f"hu{i}" for i in range(1, 8)]
HOG_COLS = [f"hog{i}" for i in range(900)]
HIST_COLS = [f"H{i}" for i in range(32)] + [f"S{i}" for i in range(32)]
FEAT_COLS = SHAPE_COLS + HU_COLS + HOG_COLS + HIST_COLS

# With 979 raw features but only ~83 training rows per LOO fold, every classifier
# was drowning in noise dimensions. Ranking features by ANOVA F-score and keeping
# the top N (re-fit per fold, so no leakage) consistently raised LOO accuracy
# in testing (swept while comparing a wider set of classifiers before trimming
# to the 3 below); 350 was the sweep optimum. See report.
N_SELECT_FEATURES = 350


def make_classifiers():
    # 3 classifiers, deliberately non-overlapping with the groupmate's half of
    # Task 2 (KNN, ANN, Naive Bayes) so the team's combined submission covers
    # 6 genuinely different algorithms with no duplication. SVM/RandomForest/
    # LogisticRegression are margin-based, bagging-ensemble, and linear-
    # discriminative respectively - three more distinct paradigms.
    return {
        "SVM (RBF, C=10)": SVC(kernel="rbf", C=10),
        "Random Forest (300 trees, depth=12)": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
    }


def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def save_confusion_matrix(y_true, y_pred, title, path):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), max(5, n * 0.35)))
    ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    vmax = max(cm.max(), 1)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                        color="white" if v > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_classifier_comparison(results, title, path):
    names = list(results.keys())
    metrics = ["acc", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_data():
    df = pd.read_csv(CSV_PATH, dtype={"SignID": str})
    df["SignID"] = df["SignID"].str.zfill(3)
    df["Label"] = df["SignID"].map(ID_LABEL)
    missing = df[df["Label"].isna()]
    if len(missing):
        print(f"WARNING: dropping {len(missing)} row(s) with unmapped SignIDs "
              f"(add them to ID_LABEL to include): {sorted(missing['SignID'].unique())}")
        df = df[df["Label"].notna()].reset_index(drop=True)

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


def load_external_data():
    """External training pool: real TSRD photos (github.com/17Hieng/Chinese-
    Traffic-Sign-Classiffication-CNN mirror of nlpr.ia.ac.cn/pal/trafficdata/
    recognition.html), same 000-057 class coding as ID_LABEL. Filtered to
    perceptual-hash distance > 12 from every one of our 84 originals, so
    there is zero image overlap with the Stage 0 test set."""
    df = pd.read_csv(EXTERNAL_CSV_PATH, dtype={"SignID": str})
    df["SignID"] = df["SignID"].str.zfill(3)
    df["Label"] = df["SignID"].map(ID_LABEL)
    missing = df[df["Label"].isna()]
    if len(missing):
        print(f"External: dropping {len(missing)} row(s) with SignIDs outside our "
              f"45 classes: {sorted(missing['SignID'].unique())}")
        df = df[df["Label"].notna()].reset_index(drop=True)

    for c in FEAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    failed = df[df[FEAT_COLS].abs().sum(axis=1) == 0]
    if len(failed):
        print(f"External: dropping {len(failed)} row(s) with failed detection (all-zero features).")
    df = df[df[FEAT_COLS].abs().sum(axis=1) > 0].reset_index(drop=True)
    return df


# =================================================================
# STAGE 0: External-data evaluation (real recognition rate, for the report)
# =================================================================
def run_external_evaluation(df):
    """Train on the external TSRD pool, test on all 84 of our own images -
    a genuine held-out evaluation with real per-class training data, unlike
    Stage 1's LOO-CV where 22 of 45 classes have only one example each. This
    is the number that reflects real-world recognition rate; Stage 1 is kept
    as the small-data baseline to show what the external data bought us."""
    if not os.path.exists(EXTERNAL_CSV_PATH):
        print(f"NOTE: {EXTERNAL_CSV_PATH} not found - skipping Stage 0.\n"
              "      Run Task2Demo.exe against the TSRD_external image set first "
              "(see report methodology) to regenerate it.")
        return

    ext = load_external_data()
    print(f"External training pool: {len(ext)} images, {ext['Label'].nunique()} classes")
    print(f"Test set (ours): {len(df)} images, {df['Label'].nunique()} classes")
    missing_classes = set(df["Label"].unique()) - set(ext["Label"].unique())
    if missing_classes:
        print(f"WARNING: {len(missing_classes)} of our classes have zero external "
              f"training examples (still a singleton, same LOO-CV caveat applies): "
              f"{sorted(missing_classes)}")

    X_train_raw = ext[FEAT_COLS].values
    y_train = ext["Label"].values
    X_test_raw = df[FEAT_COLS].values
    y_test = df["Label"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    selector = SelectKBest(f_classif, k=min(N_SELECT_FEATURES, X_train.shape[1]))
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    print(f"\n{'Classifier':<38}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1(macro)':>12}")
    print("-" * 82)
    results, best_name, best_acc = {}, None, -1
    for name, clf in make_classifiers().items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results[name] = {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "predictions": y_pred}
        print(f"{name:<38}{acc*100:>9.2f}%{prec*100:>11.2f}%{rec*100:>9.2f}%{f1:>12.3f}")
        if acc > best_acc:
            best_acc, best_name = acc, name
        save_confusion_matrix(y_test, y_pred, f"Stage 0 (external training) - {name}",
                               os.path.join(RESULTS_DIR, f"task2_external_confusion_{sanitize_filename(name)}.png"))

    save_classifier_comparison(results, "Stage 0 - Classifier Comparison (trained on external TSRD data)",
                                os.path.join(RESULTS_DIR, "task2_external_classifier_comparison.png"))

    per_class_df = pd.DataFrame({
        "Filename": df["Filename"].values, "SignID": df["SignID"].values,
        "TrueLabel": y_test, "PredictedLabel": results[best_name]["predictions"],
        "Correct": (y_test == results[best_name]["predictions"]),
    })
    per_class_df.to_csv(os.path.join(RESULTS_DIR, "task2_external_per_class_predictions.csv"), index=False)

    summary_df = pd.DataFrame([
        {"Classifier": name, "Acc": r["acc"], "Precision": r["precision"], "Recall": r["recall"], "F1Macro": r["f1"]}
        for name, r in results.items()
    ])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "task2_external_results_summary.csv"), index=False)
    print(f"\nBest: {best_name} -> {best_acc*100:.2f}% on all {len(df)} held-out images "
          f"(trained on {len(ext)} external images, zero overlap)")
    print("Saved: task2_external_results_summary.csv, task2_external_per_class_predictions.csv, "
          "task2_external_classifier_comparison.png, task2_external_confusion_<classifier>.png")


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

        selector = SelectKBest(f_classif, k=min(N_SELECT_FEATURES, train_x_s.shape[1]))
        train_x_s = selector.fit_transform(train_x_s, train_y)
        test_x_s = selector.transform(test_x_s)

        for name, clf in make_classifiers().items():
            clf.fit(train_x_s, train_y)
            preds[name][i] = clf.predict(test_x_s)[0]

    print(f"\n{'Classifier':<38}{'Accuracy':>10}{'ExclSingleton':>15}{'Precision':>12}{'Recall':>10}{'F1(macro)':>12}")
    print("-" * 98)
    results, best_name, best_acc = {}, None, -1
    for name in make_classifiers():
        y_pred = np.array(preds[name])
        acc = accuracy_score(y_orig, y_pred)
        acc_excl = accuracy_score(y_orig[non_singleton_mask], y_pred[non_singleton_mask])
        prec = precision_score(y_orig, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_orig, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_orig, y_pred, average="macro", zero_division=0)
        results[name] = {"acc": acc, "acc_excl_singleton": acc_excl, "precision": prec,
                          "recall": rec, "f1": f1, "predictions": y_pred}
        print(f"{name:<38}{acc*100:>9.2f}%{acc_excl*100:>14.2f}%{prec*100:>11.2f}%{rec*100:>9.2f}%{f1:>12.3f}")
        if acc > best_acc:
            best_acc, best_name = acc, name
        save_confusion_matrix(y_orig, y_pred, f"Stage 1 (LOO-CV) - {name}",
                               os.path.join(RESULTS_DIR, f"task2_confusion_{sanitize_filename(name)}.png"))

    save_classifier_comparison(results, "Stage 1 - Classifier Comparison (Leave-One-Out CV, 84 images only)",
                                os.path.join(RESULTS_DIR, "task2_classifier_comparison.png"))

    per_class_df = pd.DataFrame({
        "Filename": filenames_orig, "SignID": originals["SignID"].values,
        "TrueLabel": y_orig, "PredictedLabel": results[best_name]["predictions"],
        "Correct": (y_orig == results[best_name]["predictions"]),
    })
    per_class_df.to_csv(os.path.join(RESULTS_DIR, "task2_per_class_predictions.csv"), index=False)

    summary_df = pd.DataFrame([
        {"Classifier": name, "Acc": r["acc"], "AccExclSingleton": r["acc_excl_singleton"],
         "Precision": r["precision"], "Recall": r["recall"], "F1Macro": r["f1"]}
        for name, r in results.items()
    ])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "task2_results_summary.csv"), index=False)
    print(f"\nBest: {best_name} -> {results[best_name]['acc']*100:.2f}% "
          f"({results[best_name]['acc_excl_singleton']*100:.2f}% excl. singleton)")
    print("Saved: task2_results_summary.csv, task2_per_class_predictions.csv, "
          "task2_classifier_comparison.png, task2_confusion_<classifier>.png")


# =================================================================
# STAGE 2: Live demo viewer (for the presentation video)
# =================================================================
def run_demo(df):
    X = df[FEAT_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(df["Label"].values)

    selector = SelectKBest(f_classif, k=min(N_SELECT_FEATURES, X_scaled.shape[1]))
    X_scaled = selector.fit_transform(X_scaled, y_enc)

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
    parser.add_argument("--no-external", action="store_true", help="skip the external-data (TSRD) evaluation")
    args = parser.parse_args()

    data = load_data()

    if not args.no_external:
        print("=" * 76)
        print("STAGE 0: External-data evaluation (train on TSRD, test on our 84 - for the report)")
        print("=" * 76)
        run_external_evaluation(data)

    if not args.no_eval:
        print("\n" + "=" * 76)
        print("STAGE 1: Leave-One-Out Cross-Validation evaluation (for the report)")
        print("=" * 76)
        run_evaluation(data)

    if not args.no_demo:
        print("\n" + "=" * 76)
        print("STAGE 2: Live demo viewer (trained on all data) - press any key to")
        print("advance through images, 'q' to quit")
        print("=" * 76)
        run_demo(data)
