import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('fsr_time_vibrate_dataset.csv')

# Display basic info
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nDataset Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values Count:")
print(df.isna().sum())

# Visualize the data
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df['fsr'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of FSR Values')
plt.xlabel('FSR (Pressure)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(df['time'], bins=30, color='green', alpha=0.7)
plt.title('Distribution of Time Values')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(df['vibrate'], bins=3, color='red', alpha=0.7)
plt.title('Distribution of Vibration Decisions')
plt.xlabel('Vibrate (0/1)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/data_distribution.png')

# Correlation analysis
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('results/correlation_matrix.png')

# Scatter plot of FSR vs Time colored by Vibration decision
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['fsr'], df['time'], c=df['vibrate'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Vibrate')
plt.title('FSR vs Time with Vibration Decision')
plt.xlabel('FSR (Pressure)')
plt.ylabel('Time (seconds)')
plt.tight_layout()
plt.savefig('results/fsr_time_scatter.png')

# Prepare data for modeling
X = df[['fsr', 'time']]
y = df['vibrate']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Dictionary to store results
results = {}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'pipeline': pipeline,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

# Evaluate each model
print("\nEvaluating models...")
for name, model in models.items():
    print(f"\n{'-'*50}")
    print(f"Evaluating: {name}")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results[name]['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Vibrate', 'Vibrate'],
        yticklabels=['No Vibrate', 'Vibrate']
    )
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()
    
    print("\nClassification Report:")
    print(classification_report(y_test, results[name]['y_pred']))

# Find the best model based on F1 score
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]

print(f"\n{'-'*50}")
print(f"Best Model: {best_model_name}")
print(f"Best Model F1 Score: {best_model['f1']:.4f}")

# Fine-tune the best model with GridSearchCV
if best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
elif best_model_name == 'Logistic Regression':
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2', 'elasticnet', None],
        'model__solver': ['liblinear', 'saga', 'lbfgs']
    }
else:  # SVM
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__gamma': ['scale', 'auto', 0.1, 0.01]
    }

print("\nFine-tuning the best model...")
grid_search = GridSearchCV(
    best_model['pipeline'],
    param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)

tuned_acc = accuracy_score(y_test, y_pred_tuned)
tuned_prec = precision_score(y_test, y_pred_tuned)
tuned_rec = recall_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned)

print("\nTuned Model Performance:")
print(f"Accuracy: {tuned_acc:.4f}")
print(f"Precision: {tuned_prec:.4f}")
print(f"Recall: {tuned_rec:.4f}")
print(f"F1 Score: {tuned_f1:.4f}")

plt.figure(figsize=(8, 6))
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(
    cm_tuned, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['No Vibrate', 'Vibrate'],
    yticklabels=['No Vibrate', 'Vibrate']
)
plt.title(f'Confusion Matrix - Tuned {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix_tuned_model.png')
plt.close()

print("\nClassification Report for Tuned Model:")
print(classification_report(y_test, y_pred_tuned))

# Function to determine optimal FSR thresholds
def find_optimal_thresholds(model, X, y):
    if hasattr(model['model'], 'feature_importances_'):
        importances = model['model'].feature_importances_
        feature_names = ['FSR', 'Time']
        print("\nFeature Importances:")
        for feature, importance in zip(feature_names, importances):
            print(f"{feature}: {importance:.4f}")
        
        plt.figure(figsize=(10, 8))
        x_min, x_max = X['fsr'].min() - 50, X['fsr'].max() + 50
        y_min, y_max = X['time'].min() - 10, X['time'].max() + 10
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max, 2))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        scatter = plt.scatter(X['fsr'], X['time'], c=y, cmap='viridis', edgecolor='k', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Vibrate')
        plt.title('Decision Boundary for FSR and Time')
        plt.xlabel('FSR Value')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig('results/decision_boundary.png')
        plt.close()

    # Define time thresholds and FSR range
    time_thresholds = [30, 60, 120, 180, 300, 600]
    fsr_values = np.arange(X['fsr'].min(), X['fsr'].max() + 1, 1)  # Finer step size
    
    thresholds = {}
    for time in time_thresholds:
        print(f"\nAnalyzing time threshold: {time} seconds")
        predictions = []
        for fsr in fsr_values:
            test_sample = pd.DataFrame({'fsr': [fsr], 'time': [time]})
            prediction = model.predict(test_sample)[0]
            predictions.append((fsr, prediction))
        
        # Find transition from 0 to 1
        found = False
        for i in range(1, len(predictions)):
            if predictions[i-1][1] == 0 and predictions[i][1] == 1:
                thresholds[time] = predictions[i][0]
                print(f"Threshold found: FSR={thresholds[time]} for time={time}s")
                found = True
                break
        
        # If no transition, use a default or max FSR
        if not found:
            # Check if all predictions are 1 or 0
            if all(p[1] == 1 for p in predictions):
                thresholds[time] = fsr_values[0]  # Lowest FSR triggers vibration
                print(f"No transition (all vibrate): FSR={thresholds[time]} for time={time}s")
            elif all(p[1] == 0 for p in predictions):
                thresholds[time] = fsr_values[-1]  # Highest FSR as fallback
                print(f"No transition (no vibrate): FSR={thresholds[time]} for time={time}s")
            else:
                # Use median FSR from dataset as fallback
                thresholds[time] = int(X['fsr'].median())
                print(f"No clear transition: Using median FSR={thresholds[time]} for time={time}s")
    
    return thresholds

# Find and save optimal thresholds
print("\nFinding optimal threshold values...")
optimal_thresholds = find_optimal_thresholds(tuned_model, X, y)

print("\nRecommended FSR thresholds for different sitting times:")
for time, fsr in optimal_thresholds.items():
    print(f"For {time} seconds sitting time: FSR threshold = {fsr}")

# Save the model and thresholds
print("\nSaving model and thresholds...")
joblib.dump(tuned_model, 'models/seat_monitor_model.pkl')

with open('models/optimal_thresholds.txt', 'w') as f:
    f.write("Optimal FSR thresholds for different sitting times:\n")
    for time, fsr in optimal_thresholds.items():
        f.write(f"Time: {time} seconds, FSR: {fsr}\n")

print("\nTraining and evaluation complete!")
print("Model saved as: models/seat_monitor_model.pkl")
print("Thresholds saved as: models/optimal_thresholds.txt")
print("Results and visualizations saved in the results folder.")

# Example prediction function
print("\nExample prediction function:")
print("""
def predict_vibration(fsr_value, sitting_time_seconds, model):
    test_sample = pd.DataFrame({'fsr': [fsr_value], 'time': [sitting_time_seconds]})
    prediction = model.predict(test_sample)[0]
    probability = model.predict_proba(test_sample)[0][1]
    return prediction, probability

model = joblib.load('models/seat_monitor_model.pkl')
should_vibrate, probability = predict_vibration(250, 120, model)
print(f"Should vibrate: {should_vibrate}, Probability: {probability:.2f}")
""")