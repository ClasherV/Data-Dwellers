import pandas as pd

url = "https://raw.githubusercontent.com/Mansi-2/DS_journey_log/refs/heads/main/Datasets/test_scores.csv"
df = pd.read_csv(url)
df.to_csv("assessment.csv", index=False)
print("Downloaded and saved as test_attendance_highrisk_with_names.csv")
