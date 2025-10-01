
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
warnings.filterwarnings('ignore')

def ml_model():
    
    try:
        data = {
            'test_attendance': pd.read_csv("TestData/test_attendance.csv"),
            'test_fees': pd.read_csv("TestData/test_fees.csv"),
            'test_students': pd.read_csv("TestData/test_students.csv"),
            'test_assessment_physics' : pd.read_csv("TestData/test_assessment_Physics.csv"),
            'test_assessment_biology' : pd.read_csv("TestData/test_assessment_Biology.csv"),
            'test_assessment_maths' : pd.read_csv("TestData/test_assessment_Maths.csv"),
            'test_assessment_english' : pd.read_csv("TestData/test_assessment_English.csv"),
            'test_assessment_chemistry' : pd.read_csv("TestData/test_assessment_Chemistry.csv"),
            'test_assessment_geography' : pd.read_csv("TestData/test_assessment_Geography.csv"),
            'test_assessment_history' : pd.read_csv("TestData/test_assessment_History.csv"),
        }
        print("All files loaded successfully!")
    
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
    
    print("\n📊 Data shapes:")
    for key, df in data.items():
        print(f"   {key}: {df.shape}")
    
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
                if pd.notna(mark) and mark < 40:
                    failed_count += 1
            failed_attempts.append(failed_count)
    
        return failed_attempts
    
    
    # Combine test assessment data
    print("\n🔄 Combining test assessment data...")
    test_subjects = ['physics', 'biology', 'maths', 'english', 'chemistry', 'geography', 'history']
    test_assessment_combined = None
    
    for subject in test_subjects:
        key = f'test_assessment_{subject}'
        if key in data:
            df = data[key].copy()
    
    
            for col in df.columns:
                if col not in ['Roll_no', 'Name']:
                    df = df.rename(columns={col: f"{col}_{subject}"})
    
            # Calculate failed attempts for test data
            failed_attempts = calculate_failed_attempts_dynamically(df, subject)
            if failed_attempts is not None:
                df[f'Failed_Attempts_{subject}'] = failed_attempts
    
    
            if test_assessment_combined is None:
                test_assessment_combined = df
            else:
                test_assessment_combined = test_assessment_combined.merge(df, on=['Roll_no', 'Name'], how='outer')
    
            print(f"   ✅ Processed {subject}")
    
    print(f"   📈 Final test assessment shape: {test_assessment_combined.shape}")
    
    
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
        fee_features['severely_delayed'] = (fee_features['payment_delay_days'] > 30).astype(int)
    
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
    
        attendance_features['low_attendance'] = (attendance_features['attendance_percentage'] < 75).astype(int)
        attendance_features['very_low_attendance'] = (attendance_features['attendance_percentage'] < 60).astype(int)
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
        demo_features['email_domain'] = demo_features['Student_Email'].str.split('@').str[1]
    
        # Gender
        demo_features['is_female'] = (demo_features['Gender'] == 'Female').astype(int)
    
        # Parent email
        demo_features['parent_email_domain'] = demo_features['Parent_Email'].str.split('@').str[1]
    
        return demo_features[['Roll_no', 'Gender', 'age', 'contact_length', 'email_domain', 'is_female', 'parent_email_domain']]
    
    
    # Create test features
    test_academic_features = create_academic_features(test_assessment_combined)
    test_fee_features = create_fee_features(data['test_fees'])
    test_attendance_features = create_attendance_features(data['test_attendance'])
    test_demo_features = create_student_demographic_features(data['test_students'])
    
    # Merge test features
    test_merged = test_academic_features.merge(test_fee_features, on='Roll_no', how='left')
    test_merged = test_merged.merge(test_attendance_features, on='Roll_no', how='left')
    test_merged = test_merged.merge(test_demo_features, on='Roll_no', how='left')
    
    
    
    print(f"📊 Test data shape: {test_merged.shape}")
    
    
    X_test = test_merged.drop(['Roll_no'], axis=1, errors='ignore')
    
    
    # Load it back when needed
    import joblib
    calibrated_model = joblib.load("calibrated_model.pkl")
    print("✅ Model loaded back successfully")
    
    calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    
    final_test_proba = calibrated_proba
    
    # Enhanced risk categorization
    def categorize_risk_enhanced(probabilities):
        """Enhanced risk categorization with more granular levels"""
        risk_levels = []
        risk_scores = []
    
        for prob in probabilities:
            if prob >= 0.7:
                risk_levels.append('High Risk')
                risk_scores.append(3)
    
            elif prob >= 0.5:
                risk_levels.append('Medium Risk')
                risk_scores.append(2)
    
            else:
                risk_levels.append('Low Risk')
                risk_scores.append(1)
    
        return risk_levels, risk_scores
    
    risk_categories, risk_scores = categorize_risk_enhanced(final_test_proba)
    
    # Create comprehensive results
    print("\n📋 Creating results...")
    
    
    results = pd.DataFrame({
        'Roll_no': test_merged['Roll_no'],
        'Name': data['test_students']['Name'],
        'Dropout_Probability': final_test_proba,
        'Dropout_Probability_Percentage': (final_test_proba * 100).round(2),
        'Risk_Level': risk_categories,
        'Risk_Score': risk_scores
    })
    
    
    feature_columns = [
        'attendance_percentage', 'fee_paid_ratio', 'total_failed_attempts',
        'avg_all_marks', 'subjects_with_failures', 'payment_delay_days',
        'marks_below_40', 'failure_rate', 'low_attendance', 'very_low_attendance'
    ]
    
    for feature in feature_columns:
        if feature in test_merged.columns:
            results[feature] = test_merged[feature].round(3)
    
    # Add subject-wise failed attempts
    failed_subject_cols = [col for col in test_merged.columns if col.startswith('Failed_Attempts_')]
    for col in failed_subject_cols:
        subject = col.replace('Failed_Attempts_', '')
        results[f'Failed_in_{subject}'] = test_merged[col]
    
    # Add recommendations based on risk level
    def generate_recommendations(row):
        recommendations = []
    
        if row['Dropout_Probability'] > 0.7:
            if row.get('attendance_percentage', 100) < 75:
                recommendations.append("Improve attendance immediately")
            if row.get('fee_paid_ratio', 1) < 0.8:
                recommendations.append("Clear pending fees")
            if row.get('total_failed_attempts', 0) > 5:
                recommendations.append("Academic counseling required")
            if row.get('payment_delay_days', 0) > 30:
                recommendations.append("Payment discipline needed")
    
        elif row['Dropout_Probability'] > 0.4:
            if row.get('attendance_percentage', 100) < 80:
                recommendations.append("Monitor attendance regularly")
            if row.get('fee_paid_ratio', 1) < 0.9:
                recommendations.append("Ensure fee payments are timely")
            if row.get('total_failed_attempts', 0) > 2:
                recommendations.append("Focus on academic improvement")
    
        return "; ".join(recommendations) if recommendations else "No immediate action required"
    
    results['Recommendations'] = results.apply(generate_recommendations, axis=1)
    
    # Display results summary
    print("\n📊 Risk Distribution:")
    risk_distribution = results['Risk_Level'].value_counts().sort_index()
    for level, count in risk_distribution.items():
        percentage = (count / len(results)) * 100
        print(f"   {level}: {count} students ({percentage:.1f}%)")
    
    print(f"\n🎯 Average Dropout Probability: {results['Dropout_Probability'].mean():.3f}")
    
    # Save results
    output_filename = 'student_dropout_risk_analysis.csv'
    results.to_csv(output_filename, index=False)
    
    print(f"\n💾 Results saved to '{output_filename}'")
    
    return results
