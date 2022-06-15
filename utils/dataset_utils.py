from typing import Tuple, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def apply_transformer(
        dataset: pd.DataFrame,
        scaler_class,
        columns_names: List[str],
        **kwargs
):
    scaler = scaler_class(**kwargs)
    dataset[columns_names] = scaler.fit_transform(dataset[columns_names])
    return dataset

def preprocess_dataset(
        dataset: pd.DataFrame,
        std_scaler: Union[List, bool] = False,
        min_max_scaler: Union[List, bool] = False,
        one_hot_encoder: Union[List, bool] = False,
) -> pd.DataFrame:
    if std_scaler:
        dataset = apply_transformer(dataset, StandardScaler, std_scaler)
    if min_max_scaler:
        dataset = apply_transformer(dataset, MinMaxScaler, min_max_scaler)
    if one_hot_encoder:
        encoder = OneHotEncoder(sparse=False)
        transformed_regions = encoder.fit_transform(dataset[one_hot_encoder])
        dataset[encoder.categories_[0]] = transformed_regions
        dataset = dataset.drop(columns=one_hot_encoder)

    # convert gender column to boolean where male = 1 and female = 0.
    dataset["sex"] = dataset["sex"].apply(
        lambda sex: 1 if sex == "male" else 0
    )
    dataset = dataset.rename({"sex": "is_male"})

    # convert smoker column to boolean where smoker = 1 and non-smoker = 0.
    dataset["smoker"] = dataset["smoker"].apply(
        lambda sex: 1 if sex == "yes" else 0
    )

    breakpoint()

    return dataset



def load_dataset(path: str, **preprocess_args) -> Tuple:
    dataset_df = pd.read_csv(path)

    if preprocess_args:
        dataset_df = preprocess_dataset(dataset=dataset_df, **preprocess_args)

    # features are the entire columns except for charges column
    features = dataset_df.loc[:, dataset_df.columns != "charges"]
    targets = dataset_df["charges"]

    X_train, y_train, X_test, y_test = train_test_split(
        features, targets, random_state=10, test_size=0.15
    )

    return X_train, y_train, X_test, y_test


