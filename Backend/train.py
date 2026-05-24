import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from preprocess import load_and_clean_data, preprocess_x, preprocess_y
from mlflow_config import setup_mlflow, MODEL_NAME, STAGE_STAGING


def split_data(x_final, y_final, test_size=0.2, random_state=42):
    return train_test_split(x_final, y_final, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_depth", "None")
    mlflow.log_param("test_size", 0.2)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report_str)

    mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
    mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
    mlflow.log_metric("f1_score_macro", report["macro avg"]["f1-score"])
    mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
    mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_score_weighted", report["weighted avg"]["f1-score"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(report_str)
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="evaluation")
    os.unlink(f.name)

    return y_pred, accuracy


def plot_confusion_matrix(y_test, y_pred):
    cf = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plt.savefig(f.name, bbox_inches='tight')
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="evaluation")
        fname = f.name
    plt.close()
    os.unlink(fname)


def register_model(clf, run_id):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Model registered: {result.name} version {result.version}")
    return result


def save_artifacts_local(clf, scaler, encoder, label_encoder, save_dir="model/"):
    joblib.dump(clf, f"{save_dir}model.pkl")
    joblib.dump(scaler, f"{save_dir}scaler.pkl")
    joblib.dump(encoder, f"{save_dir}encoder.pkl")
    joblib.dump(label_encoder, f"{save_dir}label_encoder.pkl")
    print(f"Local artifacts saved to {save_dir}")


def run_training_pipeline(filepath, save_dir="model/"):
    setup_mlflow()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("pipeline", "full_training")
        mlflow.set_tag("model_name", MODEL_NAME)

        data = load_and_clean_data(filepath)

        y_data = data.iloc[:, 13:]
        x_data = data.iloc[:, :12]

        x_final, scaler, encoder = preprocess_x(x_data, fit=True)
        y_final, label_encoder = preprocess_y(y_data, fit=True)

        X_train, X_test, y_train, y_test = split_data(x_final, y_final)

        clf = train_model(X_train, y_train)

        mlflow.sklearn.log_model(clf, "model")

        y_pred, accuracy = evaluate_model(clf, X_test, y_test)

        plot_confusion_matrix(y_test, y_pred)

        save_artifacts_local(clf, scaler, encoder, label_encoder, save_dir)

        log_additional_artifacts(clf, save_dir)

        register_model(clf, run_id)

    return clf, accuracy, run_id


def log_additional_artifacts(clf, save_dir="model/"):
    for fname in ["scaler.pkl", "encoder.pkl", "label_encoder.pkl"]:
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path="preprocessing")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Feature Importances:\n")
        for i, imp in enumerate(clf.feature_importances_):
            f.write(f"Feature_{i}: {imp:.6f}\n")
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="model_info")
    os.unlink(f.name)
