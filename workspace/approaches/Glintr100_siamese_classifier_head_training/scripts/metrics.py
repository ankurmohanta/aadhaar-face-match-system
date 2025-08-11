import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
def evaluate_predictions(pos_csv, neg_csv, threshold=0.5):
   # Load both files
   pos_df = pd.read_csv(pos_csv)
   neg_df = pd.read_csv(neg_csv)
   # Merge
   combined = pd.concat([pos_df, neg_df], ignore_index=True)
   combined['pred_label'] = (combined['score'] >= threshold).astype(int)
   # Extract y_true and y_pred
   y_true = combined['label'].tolist()
   y_pred = combined['pred_label'].tolist()
   # Compute metrics
   precision = precision_score(y_true, y_pred)
   recall = recall_score(y_true, y_pred)
   accuracy = accuracy_score(y_true, y_pred)
   f1 = f1_score(y_true, y_pred)
   cm = confusion_matrix(y_true, y_pred)
   print("📊 Evaluation at Threshold =", threshold)
   print("✅ Precision :", round(precision, 4))
   print("✅ Recall    :", round(recall, 4))
   print("✅ Accuracy  :", round(accuracy, 4))
   print("✅ F1 Score  :", round(f1, 4))
   print("\n📉 Confusion Matrix:\n", cm)
# Example usage
if __name__ == "__main__":
   evaluate_predictions(
       pos_csv="/workspace/approaches/Glintr100_retrain_siamese_classifier/pos_pred_15-19.06.25_testpairs.csv",
       neg_csv="/workspace/approaches/Glintr100_retrain_siamese_classifier/neg_pred_15-19.06.25_testpairs.csv",
       threshold=0.7
   )
