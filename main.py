from utils.dataset_utils import load_dataset

X_train, y_train, X_test, y_test = load_dataset(
    "dataset/data.csv",
    **{
        "std_scaler": ["age", "bmi", "children"],
        # "min_max_scaler": ["age", "bmi", "children"],
        "one_hot_encoder": ["region"]
    }
)