import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import requests

def train_model():
  train_attendance = pd.read_csv("TrainData/train_attendance_with_names.csv")
  train_fees = pd.read_csv("TrainData/train_fees.csv")
  train_scores = pd.read_csv("TrainData/train_scores.csv")
  train_labels = pd.read_csv("TrainData/train_labels.csv")

  train_data = train_attendance.merge(train_fees, on="student_id").merge(train_scores, on="student_id").merge(train_labels, on="student_id")


  num_features = [
    "attendance_percent", "avg_score", "num_failed_attempts",
    "fees_paid_ratio", "assignments_submitted", "projects_completed"
]

  target = "dropout"

  x = train_data[num_features]
  y = train_data[target]

  scaler = StandardScaler()
  x_scaled = scaler.fit_transform(x)

  x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=101, stratify=y)


  smote = SMOTE(random_state=42)
  x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)


  rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=101, class_weight="balanced")
  log = LogisticRegression(max_iter=1000,random_state=101, class_weight="balanced")
  gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=101)
  scale_pos_weight = (y_train.value_counts()[0]/y_train.value_counts()[1])
  xgb=XGBClassifier(random_state=101, eval_metrics="logloss", scale_pos_weight=scale_pos_weight, use_label_encoder=False)

  ensemble = VotingClassifier(
    estimators=[("rf",rf),("log",log),("gb",gb),("xgb",xgb)],
    voting="soft"
  )

  ensemble.fit(x_train_resampled, y_train_resampled)
  y_pred = ensemble.predict(x_val)

  y_proba = ensemble.predict_proba(x_val)[:, 1]
  y_pred = (y_proba >= 0.4).astype(int)

  print("\n Ensemble Model Results:")
  print("Validation Accuracy : ",accuracy_score(y_val,y_pred))
  print("Classification Report: \n",classification_report(y_val,y_pred))

  joblib.dump(ensemble, "dropout_model.pkl")

def predict_risk(model_path="dropout_model.pkl",
                 attendance_threshold=75,
                 score_threshold=40,
                 fee_overdue_threshold=0.7,
                 max_exam_attempts=3,
                 w_attendance=0.3,
                 w_score=0.3,
                 w_fee=0.2,
                 w_attempts=0.2):
  ensemble=joblib.load(model_path)

  test_attendance = pd.read_csv("uploads/attendance.csv")
  test_fees = pd.read_csv("uploads/fees.csv")
  test_scores = pd.read_csv("uploads/assessment.csv")
  t1 = test_attendance.merge(test_fees, on="student_id")
  test_data = t1.merge(test_scores, on="student_id")

  x_test = test_data.drop(columns=["student_id","student_name","student_email"])


  numeric_features = ["attendance_percent", "avg_score", "num_failed_attempts", "fees_paid_ratio"]

  # Scale only numeric features
  scaler = StandardScaler()
  x_test_scaled = x_test.copy()
  x_test_scaled[numeric_features] = scaler.fit_transform(x_test[numeric_features])


  y_proba = ensemble.predict_proba(x_test_scaled)[:, 1] * 100  # dropout probability (%)
  y_pred = ensemble.predict(x_test_scaled)  # predicted labels (0=safe, 1=at risk)

  # Add ML predictions directly into test_data
  test_data["ML_dropout_probability"] = y_proba
  test_data["ML_prediction"] = y_pred

  # Normalize weights
  total_w = w_attendance + w_score + w_fee + w_attempts
  w_attendance, w_score, w_fee, w_attempts = (
    w_attendance / total_w,
    w_score / total_w,
    w_fee / total_w,
    w_attempts / total_w,
)

  # Rule-based risks
  test_data["attendance_risk"] = test_data["attendance_percent"].apply(
    lambda x: 1 if x < attendance_threshold else 0
)
  test_data["score_risk"] = test_data["avg_score"].apply(
    lambda x: 1 if x < score_threshold else 0
)
  test_data["fee_risk"] = test_data["fees_paid_ratio"].apply(
    lambda x: 1 if x < fee_overdue_threshold else 0
)
  test_data["attempts_risk"] = test_data["num_failed_attempts"].apply(
    lambda x: 1 if x > max_exam_attempts else 0
)

  test_data["rule_dropout_probability"] = (
    test_data["attendance_risk"] * w_attendance
    + test_data["score_risk"] * w_score
    + test_data["fee_risk"] * w_fee
    + test_data["attempts_risk"] * w_attempts
) * 100

  # Final blended probability
  test_data["final_dropout_probability"] = (
    test_data["ML_dropout_probability"] * 0.6
    + test_data["rule_dropout_probability"] * 0.4
)

  def classify_risk(prob):
    if prob >= 70:
        return "High Risk"
    elif prob >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"

  test_data["risk_level"] = test_data["final_dropout_probability"].apply(classify_risk)

  # Filter only at-risk students
  at_risk_students = test_data[test_data["risk_level"] != "Low Risk"][
    [
        "student_id",
        "student_name",
        "student_email",
        "attendance_percent",
        "avg_score",
        "fees_paid_ratio",
        "num_failed_attempts",
        "ML_dropout_probability",
        "rule_dropout_probability",
        "final_dropout_probability",
        "risk_level",
    ]
]

  at_risk_students.to_csv("at_risk_students.csv", index=False)
  print("✅ Final At-Risk Students Saved to at_risk_students.csv")

  # 🔹 Send file to Flask backend
  url = "http://127.0.0.1:5000/upload"   # your Flask app endpoint
  with open("at_risk_students.csv", "rb") as f:
    response = requests.post(url, files={"file": f})

  return at_risk_students


if __name__ == "__main__":
    # Train model once (only when needed)
    train_model()

    # Later: load and predict dynamically
    results = predict_risk(
        attendance_threshold=70,
        score_threshold=45,
        fee_overdue_threshold=0.6,
        max_exam_attempts=2,
        w_attendance=0.4,
        w_score=0.3,
        w_fee=0.2,
        w_attempts=0.1
    )

    print(results.head())