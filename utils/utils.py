import json

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
