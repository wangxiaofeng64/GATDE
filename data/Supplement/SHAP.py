import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, matthews_corrcoef
import shap
import numpy as np

# Set random seed
np.random.seed(42)

# Load data
# The data in 1.csv was obtained by merging and transposing RPPA_data_of_Lung and clinical_data_of_Lung
data = pd.read_csv('./data/Lung/1.csv')

# Data preprocessing
X = data.iloc[:, 2:]  # Assume that the features are from the third column to the last
y = data.iloc[:, 1]   # The second column is the target classification label

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model
model = SVC(kernel='linear', probability=True)  # Use linear kernel and enable probability estimates
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average='macro')  # Or 'binary' if it's a binary classification problem
precision = precision_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
mcc = matthews_corrcoef(y_test, predictions)

# Calculate specificity
cm = confusion_matrix(y_test, predictions)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Applies to binary classification, calculating specificity for class 0

# Print performance metrics
print("Accuracy:", accuracy)
print("Recall (Macro-average):", recall)
print("Precision (Weighted-average):", precision)
print("F1 Score (Weighted-average):", f1)
print("Matthews Correlation Coefficient:", mcc)
print("Specificity:", specificity)
print("Classification Report:\n", classification_report(y_test, predictions))

# Optimize SVM hyperparameters using Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Output the best parameters
print("Best parameters found: ", grid.best_params_)

# Make predictions using the best parameters
grid_predictions = grid.predict(X_test)

# Output classification report
print(classification_report(y_test, grid_predictions))

# Retrain SVM model using the best parameters
best_svm = SVC(C=10, gamma=0.01, kernel='rbf', probability=True)
best_svm.fit(X_train, y_train)

# Predict and evaluate accuracy
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print(classification_report(y_test, y_pred))

# Determine the number of samples to extract
sample_size = min(100, X_test.shape[0])
# Randomly select a subset of samples from the test set
sample_indices = np.random.choice(X_test.shape[0], size=sample_size, replace=False)
X_test_sample = X_test[sample_indices]
y_test_sample = y_test.iloc[sample_indices]

# Use SHAP values to evaluate feature importance
explainer = shap.KernelExplainer(best_svm.predict_proba, X_train)
shap_values = explainer.shap_values(X_test_sample, nsamples=200)

# Check the structure of shap_values
print("shap_values shape:", np.array(shap_values).shape)

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# Print the shape of shap_values for each class
for i, shap_val in enumerate(shap_values):
    print(f"shap_values for class {i} shape:", shap_val.shape)
    shap_values_df = pd.DataFrame(shap_val, columns=X.columns)
    shap_values_df.to_csv(f'shap_values_class_{i}.csv', index=False)

# Plot feature importance (commented out for now)
#shap.summary_plot(shap_values[1], X_test_sample, feature_names=X.columns.tolist())

# Print the mean absolute SHAP values for each feature to identify the most contributing features
shap_importance = np.mean(np.abs(shap_values[1]), axis=0)
feature_importance = pd.DataFrame(list(zip(X.columns, shap_importance)), columns=['Feature', 'SHAP Importance'])
feature_importance = feature_importance.sort_values(by='SHAP Importance', ascending=False)
print(feature_importance)
# Save feature_importance to a CSV file
feature_importance.to_csv('feature_importance_2.csv', index=False)
