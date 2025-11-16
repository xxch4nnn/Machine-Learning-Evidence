import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from Machine_Learning_Course.Code.Data_Pipeline.PreparePianoMotionDataset import extract_features

def train_and_evaluate(features_csv):
    """
    Trains and evaluates SVM and Random Forest classifiers on the processed data.

    Args:
        features_csv (str): The path to the CSV file containing the extracted features.
    """
    # Generate features if the file doesn't exist
    if not os.path.exists(features_csv):
        print(f"Features file not found at {features_csv}. Running data preparation script...")
        extract_features("Machine_Learning_Course/Data/PianoMotion10M", features_csv)
        print("Data preparation complete.")

    # Load the data
    df = pd.read_csv(features_csv)
    X = df[['depth_feature', 'tip_to_dip_distance',
            'fingertip_to_wrist_distance', 'fingertip_to_palm_center_distance',
            'tip_x', 'tip_y', 'tip_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'relative_velocity_x', 'relative_velocity_y', 'relative_velocity_z',
            'avg_velocity_x', 'avg_velocity_y', 'avg_velocity_z',
            'avg_acceleration_x', 'avg_acceleration_y', 'avg_acceleration_z']]
    y = df['is_press']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Support Vector Machine (SVM) ---
    print("--- Training Support Vector Machine (SVM) ---")
    svm = SVC()
    param_dist_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    random_search_svm = RandomizedSearchCV(svm, param_distributions=param_dist_svm, n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)
    random_search_svm.fit(X_train, y_train)

    print("Best SVM parameters found: ", random_search_svm.best_params_)
    best_svm = random_search_svm.best_estimator_

    # Evaluate SVM
    y_pred_svm = best_svm.predict(X_test)
    evaluate_model("SVM", best_svm, y_test, y_pred_svm, X_test)


    # --- Random Forest ---
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier()
    param_dist_rf = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)
    random_search_rf.fit(X_train, y_train)

    print("Best Random Forest parameters found: ", random_search_rf.best_params_)
    best_rf = random_search_rf.best_estimator_

    # Evaluate Random Forest
    y_pred_rf = best_rf.predict(X_test)
    evaluate_model("Random Forest", best_rf, y_test, y_pred_rf, X_test)


def evaluate_model(model_name, model, y_test, y_pred, X_test):
    """
    Evaluates a model and prints the performance metrics.
    """
    print(f"\n--- {model_name} Performance ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1-score: {f1_score(y_test, y_pred)}")

    # Calculate FPS
    start_time = time.time()
    for i in range(len(X_test)):
        model.predict(X_test.iloc[[i]])
    end_time = time.time()
    fps = len(X_test) / (end_time - start_time)
    print(f"Frames Per Second (FPS): {fps}")


if __name__ == "__main__":
    FEATURES_CSV = "Machine_Learning_Course/Data/PianoMotion10M/features.csv"
    train_and_evaluate(FEATURES_CSV)
