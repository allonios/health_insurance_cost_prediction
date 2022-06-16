import json
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from json_encoders.numpy_json_encoder import NumpyArrayEncoder
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve as lc
from sklearn.model_selection import validation_curve as vc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from utils.plot_utils import  plot_grid_search_results, plot_learning_curve, plot_validation_curve
from sklearn.base import BaseEstimator
from IPython.display import display
import pandas as pd

class ModelSummary:
    def __init__(
        self,
        name,
        mean_squared_error,
        r2_score,
        grid_search_summary,
        validation_curve_data,
        learning_curve_data,
        model=None
    ):
        self.mean_squared_error = mean_squared_error
        self.r2_score = r2_score
        self.grid_search_summary = grid_search_summary
        self.validation_curve_data = validation_curve_data
        self.learning_curve_data = learning_curve_data
        self.name = name
        self.model = model

    @classmethod
    def from_json_dict(cls, name, json_dict):
        return ModelSummary(
            name=name,
            mean_squared_error=json_dict["mean_squared_error"],
            r2_score=json_dict["r2_score"],
            grid_search_summary=json_dict["grid_search_summary"],
            validation_curve_data=json_dict["validation_curve_data"],
            learning_curve_data=json_dict["learning_curve_data"],
        )

    def to_json(self, path):
        obj = self.__dict__.copy()
        obj.pop("name")
        obj.pop("model")
        # if obj["validation_curve_data"]:
        #     obj["validation_curve_data"]["param_range"] = obj["validation_curve_data"]["param_range"].tolist()
        #     obj["validation_curve_data"]["val_score"] = obj["validation_curve_data"]["val_score"].tolist()
        #     obj["validation_curve_data"]["train_score"] = obj["validation_curve_data"]["train_score"].tolist()

        # if obj["learning_curve_data"]:
        #     obj["learning_curve_data"]["train_lc"] = obj["learning_curve_data"]["train_lc"].tolist()
        #     obj["learning_curve_data"]["val_lc"] = obj["learning_curve_data"]["val_lc"].tolist()
        #     obj["learning_curve_data"]["N"] = obj["learning_curve_data"]["N"].tolist()

        with open(path, "r") as file:
            content = json.load(file)

        with open(path, "w") as file:
            content[self.name] = obj
            json.dump(content, file, indent=4, cls=NumpyArrayEncoder)

    def display(self):
        df = pd.DataFrame(data={
            " ": ["Train", "Test"],
            "MSE": [self.mean_squared_error["train"], self.mean_squared_error["test"]],
            "R2 Score": [self.r2_score["train"], self.r2_score["test"]]
        })
        print("Model Result:")
        display(df)
        # print("MSE:")
        # print(self.mean_squared_error)
        # print("R2:")
        # print(self.r2_score)
        if self.grid_search_summary:
            print("Grid Search Best Params:")
            best_params = self.grid_search_summary["best_params"]
            best_params_df = pd.DataFrame({"param": list(best_params), "values": list(map(lambda x: best_params[x],best_params))})
            display(best_params_df)
            # print("Grid Search:")
            # print(self.grid_search_summary)
        
        curves = [self.validation_curve_data,
                  self.grid_search_summary, self.learning_curve_data]

        curves = list(
            filter(
                lambda curve: curve, curves
            )
        )

        current_ax_index = 0
        if self.validation_curve_data:
            plot_validation_curve(
                plt,
                self.validation_curve_data["param_range"],
                self.validation_curve_data["train_score"],
                self.validation_curve_data["val_score"],
            )
            current_ax_index += 1
            plt.show()

        if self.learning_curve_data:
            plot_learning_curve(
                plt,
                self.learning_curve_data["N"],
                self.learning_curve_data["train_lc"],
                self.learning_curve_data["val_lc"],
            )
            current_ax_index += 1
            plt.show()

        if self.grid_search_summary:
            plot_grid_search_results(
                plt,
                self.grid_search_summary["cv_results"],
                self.grid_search_summary["best_params"],
                self.grid_search_summary["param_grid"],
            )
            current_ax_index += 1
            plt.show()

def generate_model_summary(
    name: str,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    polynomial_features: Dict = None,
    select_percentile: Dict = None,
    select_from_model: Dict = None,
    rfe: Dict = None,

    validation_curve: Dict = None,
    grid_search: Dict = None,
    learning_curve: Dict = None

):
    pipline_elements = []

    if isinstance(polynomial_features, dict):
        pipline_elements.append(
            PolynomialFeatures(**polynomial_features)
        )

    if isinstance(select_percentile, dict):
        pipline_elements.append(
            SelectPercentile(**select_percentile)
        )

    if isinstance(select_from_model, dict):
        if select_from_model["estimator"]:
            pipline_elements.append(
                SelectFromModel(**select_from_model)
            )
        else:
            raise ValueError("select_from_model must contain estimator")

    if isinstance(rfe, dict):
        if rfe["estimator"]:
            pipline_elements.append(
                RFE(**rfe)
            )
        else:
            raise ValueError("rfe must contain estimator")

    pipline_elements.append(model)

    pipeline = make_pipeline(*pipline_elements)

    grid = None
    grid_search_summary = None
    if isinstance(grid_search, dict):
        if(grid_search["param_grid"]):
            grid = GridSearchCV(pipeline, cv=2, **grid_search, return_train_score=True)
        else:
            raise ValueError("grid_search must contain param_grid")

    if grid:
        grid.fit(X_train, y_train)
        y_pred_test = grid.predict(X_test)
        y_pred_train = grid.predict(X_train)

        grid_search_summary = {
            "best_params": grid.best_params_,
            "cv_results": grid.cv_results_,
            "param_grid": grid.param_grid
        }
        model = grid.best_estimator_

    else:
        pipeline.fit(X_train, y_train)
        y_pred_test = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)

        model = pipeline

    validation_curve_data = None
    if isinstance(validation_curve, dict):
        if (
            "param_name" in validation_curve
            and "param_range" in validation_curve
        ):
            val_train_score, val_score = vc(
                model, X_train, y_train,
                **validation_curve
            )
            validation_curve_data = {
                "train_score": val_train_score,
                "val_score": val_score,
                "param_range": validation_curve.get("param_range", False),
            }
        else:
            raise ValueError(
                "validation_curve must contain param_name and param_range")

    learning_curve_data = None
    if isinstance(learning_curve, dict):
        N, train_lc, val_lc = lc(model, X_train, y_train, **learning_curve)
        learning_curve_data = {"N": N, "train_lc": train_lc, "val_lc": val_lc}

    return ModelSummary(
        name=name,
        mean_squared_error={
            "test":   float(mean_squared_error(y_true=y_test, y_pred=y_pred_test)),
            "train": float(mean_squared_error(y_true=y_train, y_pred=y_pred_train)),
        },
        r2_score={
            "test":   float(r2_score(y_true=y_test, y_pred=y_pred_test)),
            "train": float(r2_score(y_true=y_train, y_pred=y_pred_train)),
        },
        grid_search_summary=grid_search_summary,
        validation_curve_data=validation_curve_data,
        learning_curve_data=learning_curve_data,
        model=model
    )
