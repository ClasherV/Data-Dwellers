from TestingModel import ml_model
results = ml_model()
output_filename = 'student_dropout_risk_analysis.csv'
results.to_csv(output_filename, index=False)