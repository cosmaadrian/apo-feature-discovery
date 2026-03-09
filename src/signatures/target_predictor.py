import dspy
from typing import List, Literal, Optional
from .utils import get_python_type_from_field
from pydantic import Field, BaseModel
from typing_extensions import TypedDict


def create_feature_prediction_signature(features_spec, dataset_name, target_variable, description, possible_classes):
    class_attrs = {"__doc__": f"Predict the target variable '{target_variable}' from the extracted features. Dataset: {dataset_name}. Description: {description}", "__annotations__": {}}

    for feature in features_spec:
        field_name = feature.name
        try:
            py_type = get_python_type_from_field(feature)
        except TypeError as e:
            # print(f"Skipping field '{field_name}' due to error: {e}")
            continue

        dspy_field_instance = dspy.InputField(desc = feature.description)

        class_attrs[field_name] = dspy_field_instance
        class_attrs["__annotations__"][field_name] = py_type

    class_attrs["target_variable"] = dspy.OutputField(desc = f"The predicted value for the target variable '{target_variable}'.")
    class_attrs["__annotations__"]["target_variable"] = Literal[tuple(possible_classes)]

    FeaturePredictionSignature = type("FeaturePredictionSignature", (dspy.Signature, ), class_attrs)

    return FeaturePredictionSignature


class PredictorModule(dspy.Module):

    def __init__(self, args, dataset_name, task_description, target_variable, possible_classes):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.target_variable = target_variable
        self.task_description = task_description
        self.possible_classes = possible_classes

    def forward(self, feature_spec, feature_values):
        signature = create_feature_prediction_signature(feature_spec, dataset_name = self.dataset_name, target_variable = self.target_variable, description = self.task_description, possible_classes = self.possible_classes)
        predictor = dspy.Predict(signature)

        result = predictor(**{feature.name: getattr(feature_values, feature.name) for feature in feature_spec})

        prediction = dspy.Prediction(target_variable = result.target_variable)
        return prediction


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
