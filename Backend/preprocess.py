import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import json
import tempfile
import os


def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)

    mlflow.log_param("dataset_shape", str(data.shape))
    mlflow.log_param("num_rows", len(data))
    mlflow.log_param("num_columns", len(data.columns))
    mlflow.log_param("column_names", list(data.columns))

    null_counts = data.isna().sum().to_dict()
    null_pcts = (data.isna().sum() / len(data) * 100).to_dict()
    mlflow.log_param("null_counts", str(null_counts))
    mlflow.log_param("null_percentages", str({k: round(v, 2) for k, v in null_pcts.items()}))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data.dtypes.astype(str).to_dict(), f)
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="data_info")
    os.unlink(f.name)

    num_imputer = SimpleImputer(strategy='mean')
    data[['daily_travel_time', 'nearby_industries', 'years_in_location']] = num_imputer.fit_transform(
        data[['daily_travel_time', 'nearby_industries', 'years_in_location']]
    )

    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[['vehicle_ownership', 'green_space_access', 'work_location_type', 'awareness_level']] = cat_imputer.fit_transform(
        data[['vehicle_ownership', 'green_space_access', 'work_location_type', 'awareness_level']]
    )

    mlflow.log_param("num_imputer_strategy", "mean")
    mlflow.log_param("cat_imputer_strategy", "most_frequent")
    mlflow.log_param("imputed_numeric_cols", ['daily_travel_time', 'nearby_industries', 'years_in_location'])
    mlflow.log_param("imputed_categorical_cols", ['vehicle_ownership', 'green_space_access', 'work_location_type', 'awareness_level'])

    return data


def preprocess_x(data, scaler=None, encoder=None, fit=False):
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = data.select_dtypes(include=['object']).columns
    bool_cols = data.select_dtypes(include=['bool']).columns

    if fit:
        mlflow.log_param("num_features", len(num_cols))
        mlflow.log_param("num_feature_names", list(num_cols))
        mlflow.log_param("cat_features_before_encoding", list(cat_cols))
        mlflow.log_param("bool_features", list(bool_cols))
        mlflow.log_param("scaler_type", "StandardScaler")
        mlflow.log_param("encoder_type", "OneHotEncoder")

        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])

        encoder = OneHotEncoder(sparse_output=False)
        x_encoded_array = encoder.fit_transform(data[cat_cols])
        x_encoded_cols = encoder.get_feature_names_out(cat_cols)
        x_encoded_df = pd.DataFrame(x_encoded_array, columns=x_encoded_cols, index=data.index)

        x_final = pd.concat([data[num_cols], x_encoded_df, data[bool_cols]], axis=1)

        mlflow.log_param("encoded_feature_names", list(x_encoded_cols))
        mlflow.log_param("final_feature_count", x_final.shape[1])
        mlflow.log_param("final_feature_names", list(x_final.columns))

        return x_final, scaler, encoder
    else:
        data[num_cols] = scaler.transform(data[num_cols])

        x_encoded_array = encoder.transform(data[cat_cols])
        x_encoded_cols = encoder.get_feature_names_out(cat_cols)
        x_encoded_df = pd.DataFrame(x_encoded_array, columns=x_encoded_cols, index=data.index)

        x_final = pd.concat([data[num_cols], x_encoded_df, data[bool_cols]], axis=1)
        return x_final


def preprocess_y(data, fit=False, label_encoder=None):
    y = data[['risk_category']]
    y_encoded = pd.get_dummies(y, columns=['risk_category'])
    y_encoded["risk_category"] = y_encoded.idxmax(axis=1)

    if fit:
        mlflow.log_param("y_target_column", "risk_category")
        mlflow.log_param("y_original_categories", list(y["risk_category"].unique()))
        mlflow.log_param("y_encoded_classes", list(y_encoded["risk_category"].unique()))

        le = LabelEncoder()
        y_final = le.fit_transform(y_encoded["risk_category"])
        mlflow.log_param("y_label_mapping", str(dict(zip(le.classes_, range(len(le.classes_))))))
        return y_final, le
    else:
        y_final = label_encoder.transform(y_encoded["risk_category"])
        return y_final
