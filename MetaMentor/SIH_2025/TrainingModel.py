import requests
import io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import warnings
from Dashboard import config
warnings.filterwarnings('ignore')

try:
    data = {
        'train_attendance': pd.read_csv("TrainData/train_attendance.csv"),
        'train_fees': pd.read_csv("TrainData/train_fees.csv"),
        'train_students': pd.read_csv("TrainData/train_students.csv"),
        'train_assessment_physics' : pd.read_csv("TrainData/train_assessment/train_assessment_Physics.csv"),
        'train_assessment_biology' : pd.read_csv("TrainData/train_assessment/train_assessment_Biology.csv"),
        'train_assessment_maths' : pd.read_csv("TrainData/train_assessment/train_assessment_Maths (1).csv"),
        'train_assessment_english' : pd.read_csv("TrainData/train_assessment/train_assessment_English.csv"),
        'train_assessment_chemistry' : pd.read_csv("TrainData/train_assessment/train_assessment_Chemistry.csv"),
        'train_assessment_geography' : pd.read_csv("TrainData/train_assessment/train_assessment_Geography.csv"),
        'train_assessment_history' : pd.read_csv("TrainData/train_assessment/train_assessment_History.csv"),
    }
    print("All files loaded successfully!")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")

# Function to calculate failed attempts dynamically
def calculate_failed_attempts_dynamically(df, subject_name):
    """Calculate failed attempts for a subject based on available tests"""
    mst_cols = [col for col in df.columns if col.startswith('MST')]

    if not mst_cols:
        return None

    failed_attempts = []
    for idx, row in df.iterrows():
        failed_count = 0
        for mst_col in mst_cols:
            mark = row[mst_col]
            if pd.notna(mark) and mark < config['score_threshold']:
                failed_count += 1
        failed_attempts.append(failed_count)

    return failed_attempts

# Combine train assessment data
print("\n🔄 Combining train assessment data...")
train_subjects = ['physics', 'biology', 'maths', 'english', 'chemistry', 'geography', 'history']
train_assessment_combined = None

for subject in train_subjects:
    key = f'train_assessment_{subject}'
    if key in data:
        df = data[key].copy()


        for col in df.columns:
            if col not in ['Roll_no', 'Name']:
                df = df.rename(columns={col: f"{col}_{subject}"})

        # Calculate failed attempts for train data
        failed_attempts = calculate_failed_attempts_dynamically(df, subject)
        if failed_attempts is not None:
            df[f'Failed_Attempts_{subject}'] = failed_attempts

        # Calculate best of two for train data
        # best_of_two = calculate_best_of_two(df, subject)
        # if best_of_two is not None:
        #     df[f'Best_of_2_{subject}'] = best_of_two

        if train_assessment_combined is None:
            train_assessment_combined = df
        else:
            train_assessment_combined = train_assessment_combined.merge(df, on=['Roll_no', 'Name'], how='outer')

        print(f"   ✅ Processed {subject}")

print(f"   📈 Final train assessment shape: {train_assessment_combined.shape}")


def create_academic_features(df, num_subjects=7):
    """Create comprehensive academic features"""
    features_df = pd.DataFrame()
    features_df['Roll_no'] = df['Roll_no']

    # Overall academic performance
    mst_cols = [col for col in df.columns if col.startswith('MST')]
    if mst_cols:
        features_df['avg_all_marks'] = df[mst_cols].mean(axis=1)
        features_df['min_all_marks'] = df[mst_cols].min(axis=1)
        features_df['max_all_marks'] = df[mst_cols].max(axis=1)
        features_df['std_all_marks'] = df[mst_cols].std(axis=1)
        features_df['marks_below_40'] = (df[mst_cols] < 40).sum(axis=1)
        features_df['marks_below_60'] = (df[mst_cols] < 60).sum(axis=1)
        features_df['consistency_score'] = 1 / (1 + features_df['std_all_marks'])

    # Failed attempts
    failed_cols = [col for col in df.columns if col.startswith('Failed_Attempts')]
    if failed_cols:
        features_df['total_failed_attempts'] = df[failed_cols].sum(axis=1)
        features_df['subjects_with_failures'] = (df[failed_cols] > 0).sum(axis=1)
        features_df['max_failed_in_subject'] = df[failed_cols].max(axis=1)
        features_df['failure_rate'] = features_df['total_failed_attempts'] / (len(failed_cols) * 3)  # 3 tests per subject

    # Best of two
    best_of_two_cols = [col for col in df.columns if col.startswith('Best_of_2')]
    if best_of_two_cols:
        features_df['avg_best_of_two'] = df[best_of_two_cols].mean(axis=1)
        features_df['min_best_of_two'] = df[best_of_two_cols].min(axis=1)
        features_df['best_of_two_below_40'] = (df[best_of_two_cols] < 40).sum(axis=1)
        features_df['best_of_two_above_80'] = (df[best_of_two_cols] > 80).sum(axis=1)

    return features_df

def create_fee_features(fees_df):
    """Create features from fee payment data"""
    fee_features = fees_df.copy()

    # fee payment metrics
    fee_features['fee_paid_ratio'] = fee_features['Paid Fee'] / fee_features['Total Fee']
    fee_features['fee_balance'] = fee_features['Total Fee'] - fee_features['Paid Fee']
    fee_features['fee_balance_ratio'] = fee_features['fee_balance'] / fee_features['Total Fee']

    # Payment timeliness
    fee_features['Payment Date'] = pd.to_datetime(fee_features['Payment Date'], errors='coerce')
    fee_features['Last Date'] = pd.to_datetime(fee_features['Last Date'], errors='coerce')
    fee_features['payment_delay_days'] = (fee_features['Payment Date'] - fee_features['Last Date']).dt.days
    fee_features['payment_on_time'] = (fee_features['payment_delay_days'] <= 0).astype(int)
    fee_features['payment_delayed'] = (fee_features['payment_delay_days'] > 0).astype(int)
    fee_features['severely_delayed'] = (fee_features['payment_delay_days'] >config['fee_overdue_days']).astype(int)

    return fee_features[['Roll_no', 'fee_paid_ratio', 'fee_balance', 'fee_balance_ratio',
                       'payment_delay_days', 'payment_on_time', 'payment_delayed', 'severely_delayed']]

def create_attendance_features(attendance_df):
    """Create features from attendance data"""
    attendance_features = attendance_df.copy()

    if 'Monthly Attendance %' in attendance_df.columns:
        attendance_features['attendance_percentage'] = attendance_df['Monthly Attendance %']
    else:
        date_columns = [col for col in attendance_df.columns if any(char.isdigit() for char in col)]
        if date_columns:
            def calculate_attendance_percentage(row):
                present_days = sum(1 for col in date_columns if row.get(col) == 'P')
                total_days = len(date_columns)
                return (present_days / total_days) * 100 if total_days > 0 else 0

            attendance_features['attendance_percentage'] = attendance_df.apply(calculate_attendance_percentage, axis=1)
        else:
            attendance_features['attendance_percentage'] = 100

    attendance_features['low_attendance'] = (attendance_features['attendance_percentage'] < config['att_threshold']).astype(int)
    attendance_features['very_low_attendance'] = (attendance_features['attendance_percentage'] < config['att_threshold']-15).astype(int)
    attendance_features['attendance_trend'] = attendance_features['attendance_percentage'] / 100

    return attendance_features[['Roll_no', 'attendance_percentage', 'low_attendance', 'very_low_attendance', 'attendance_trend']]

def create_student_demographic_features(students_df):
    """Create features from student demographic data"""
    demo_features = students_df.copy()

    # Age calculation
    demo_features['DOB'] = pd.to_datetime(demo_features['DOB'], errors='coerce')
    demo_features['age'] = (pd.to_datetime('today') - demo_features['DOB']).dt.days // 365

    # Contact number
    demo_features['contact_length'] = demo_features['Contact No.'].astype(str).str.len()

    # Email
    demo_features['email_domain'] = demo_features['Student Email'].str.split('@').str[1]

    # Gender
    demo_features['is_female'] = (demo_features['Gender'] == 'Female').astype(int)

    # Parent email
    demo_features['parent_email_domain'] = demo_features['Parent Email'].str.split('@').str[1]

    return demo_features[['Roll_no', 'Gender', 'age', 'contact_length', 'email_domain', 'is_female', 'parent_email_domain']]

def create_target_variable(df, base_dropout_rate=0.15):
    """Create synthetic target variable for training"""
    np.random.seed(42)


    dropout_proba = np.ones(len(df)) * base_dropout_rate

    feature_weights = {
        'total_failed_attempts': 0.3,
        'attendance_percentage': 0.2,
        'fee_paid_ratio': 0.2,
        f'marks_below_{config['score_threshold']*100}': 0.15,
        'payment_delay_days': 0.1,
        'std_all_marks': 0.05
    }

    for feature, weight in feature_weights.items():
        if feature in df.columns:
            if feature == 'attendance_percentage':
                feature_weight = (100 - df[feature]) / 100
            elif feature == 'fee_paid_ratio':
                feature_weight = (1 - df[feature]).fillna(1)
            elif feature in ['payment_delay_days', 'std_all_marks']:
                feature_weight = df[feature] / max(df[feature].max(), 1)
            else:
                feature_weight = df[feature] / max(df[feature].max(), 1)

            dropout_proba += feature_weight * weight


    dropout_proba = np.clip(dropout_proba, 0.05, 0.95)


    target = (np.random.random(len(df)) < dropout_proba).astype(int)

    return target, dropout_proba

# Create features
print("\n🔧 Creating features...")
train_academic_features = create_academic_features(train_assessment_combined)
train_fee_features = create_fee_features(data['train_fees'])
train_attendance_features = create_attendance_features(data['train_attendance'])
train_demo_features = create_student_demographic_features(data['train_students'])

# Merge train features
train_merged = train_academic_features.merge(train_fee_features, on='Roll_no', how='left')
train_merged = train_merged.merge(train_attendance_features, on='Roll_no', how='left')
train_merged = train_merged.merge(train_demo_features, on='Roll_no', how='left')


# Create target variable
y_train, dropout_proba_train = create_target_variable(train_merged)
train_merged['dropout_risk'] = y_train
train_merged['dropout_probability'] = dropout_proba_train

print(f"📊 Train data shape: {train_merged.shape}")


X_train = train_merged.drop(['Roll_no', 'dropout_risk', 'dropout_probability'], axis=1, errors='ignore')


categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"🔤 Categorical columns: {len(categorical_cols)}")
print(f"🔢 Numerical columns: {len(numerical_cols)}")


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"⚖️ Class weights: {class_weight_dict}")


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])


print("\n🏗️ Building Ensemble Models...")


rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight_dict,
        random_state=42
    ))
])

gb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ))
])

xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        scale_pos_weight=class_weight_dict[1],
        random_state=42,
        eval_metric='logloss'
    ))
])

svm_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(
        probability=True,
        class_weight=class_weight_dict,
        random_state=42
    ))
])


voting_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight_dict)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=200, random_state=42, scale_pos_weight=class_weight_dict[1]))
        ],
        voting='soft'
    ))
])


stacking_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight_dict)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=200, random_state=42, scale_pos_weight=class_weight_dict[1]))
        ],
        final_estimator=LogisticRegression(class_weight=class_weight_dict, random_state=42),
        cv=5
    ))
])


models = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'XGBoost': xgb_model,
    'SVM': svm_model,
    'Voting Classifier': voting_clf,
    'Stacking Classifier': stacking_clf
}

# Train and evaluate all models
print("\n🚀 Training Ensemble Models...")
model_performance = {}
predictions = {}

for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train, y_train)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    model_performance[name] = {
        'cv_roc_auc_mean': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std()
    }

    print(f"   ✅ {name} - CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Use the best model based on cross-validation
best_model_name = max(model_performance, key=lambda x: model_performance[x]['cv_roc_auc_mean'])
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📈 Best CV ROC-AUC: {model_performance[best_model_name]['cv_roc_auc_mean']:.4f}")



calibrated_model = CalibratedClassifierCV(best_model, cv=5, method='isotonic')
calibrated_model.fit(X_train, y_train)


import pickle

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("calibrated_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

