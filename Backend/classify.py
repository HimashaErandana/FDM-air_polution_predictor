import mlflow
from mlflow_config import setup_mlflow, MODEL_NAME, STAGE_PRODUCTION


setup_mlflow()


def load_production_model():
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE_PRODUCTION}")
        print(f"Loaded model: {MODEL_NAME} ({STAGE_PRODUCTION})")
        return model
    except Exception:
        print(f"No production model found, trying Staging...")
        try:
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
            print(f"Loaded model: {MODEL_NAME} (Staging)")
            return model
        except Exception as e:
            print(f"Failed to load from registry: {e}")
            import joblib
            model = joblib.load("model/model.pkl")
            print("Falling back to local model.pkl")
            return model


model = load_production_model()


def classify(data):
    prediction = model.predict(data)
    res = ''
    if prediction[0] == 0:
        res = 'risk_high'
    elif prediction[0] == 1:
        res = 'risk_low'
    elif prediction[0] == 2:
        res = 'risk_medium'

    return res
