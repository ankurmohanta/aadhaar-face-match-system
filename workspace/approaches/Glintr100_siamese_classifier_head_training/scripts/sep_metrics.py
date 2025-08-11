import pandas as pd

from sklearn.metrics import precision_score, recall_score, accuracy_score

def evaluate(df, true_label, threshold):

    if true_label == 1:

        preds = (df['score'] >= threshold).astype(int)

        labels = [1] * len(df)

    else:

        preds = (df['score'] < threshold).astype(int)

        labels = [1] * len(df)  # still treated as "positives" for metric calc on this class

    precision = precision_score(labels, preds, zero_division=0)

    recall = recall_score(labels, preds, zero_division=0)

    accuracy = accuracy_score(labels, preds)

    return precision, recall, accuracy

# === Load both CSVs ===

pos_csv = "/workspace/approaches/Glintr100_retrain_siamese_classifier/pos_pred_15-19.06.25_testpairs.csv"  # <- update with your actual path

neg_csv = "/workspace/approaches/Glintr100_retrain_siamese_classifier/neg_pred_15-19.06.25_testpairs.csv"  # <- update with your actual path

pos_df = pd.read_csv(pos_csv)

neg_df = pd.read_csv(neg_csv)

threshold = 0.5

# === Positive Pair Metrics ===

pos_precision, pos_recall, pos_accuracy = evaluate(pos_df, true_label=1, threshold=threshold)

# === Negative Pair Metrics ===

neg_precision, neg_recall, neg_accuracy = evaluate(neg_df, true_label=0, threshold=threshold)

# === Display Results ===

print(f"✅ Positive Pairs @ Threshold = {threshold}")

print(f"Precision: {pos_precision:.4f}")

print(f"Recall:    {pos_recall:.4f}")

print(f"Accuracy:  {pos_accuracy:.4f}\n")

print(f"❌ Negative Pairs @ Threshold = {threshold}")

print(f"Precision: {neg_precision:.4f}")

print(f"Recall:    {neg_recall:.4f}")

print(f"Accuracy:  {neg_accuracy:.4f}")
 