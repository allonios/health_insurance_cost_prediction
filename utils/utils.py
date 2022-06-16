import json
import math
from typing import List

from models.model_builder import ModelSummary


def import_from_json(path: str):
    with open(path) as file:
        content = json.load(file)

    return list(
        map(
            lambda model_name: ModelSummary.from_json_dict(
                model_name, content[model_name]),
            content
        )
    )

def get_best_model(models_summaries: List[ModelSummary]):
    best_mse_model = None
    best_r2_model = None
    best_mse = math.inf
    best_r2 = -math.inf
    for model_summary in models_summaries:
        if model_summary.mean_squared_error < best_mse:
            best_mse_model = model_summary
            best_mse = model_summary.mean_squared_error
        if model_summary.r2_score > best_r2:
            best_r2_model = model_summary

            best_r2 = model_summary.r2_score
    return best_mse_model, best_r2_model
