import dspy
from typing import List, Optional, TypedDict
from .utils import Feature


def create_interpretability_signature_class(proposed_features) -> type:
    """
    Creates a dspy.Signature class for feature feedback.

    Returns:
        A new class that inherits from dspy.Signature.
    """
    class_name = "_InterpretabilitiySignature"
    docstring = _InterpretabilitiySignature.__doc__

    class_attrs = {"__doc__": docstring, "__annotations__": {}}

    for field_name, field_type in _InterpretabilitiySignature.__annotations__.items():
        description = _InterpretabilitiySignature.__dict__['__pydantic_fields__'][field_name].json_schema_extra['desc']

        if _InterpretabilitiySignature.__dict__['__pydantic_fields__'][field_name].json_schema_extra['__dspy_field_type'] == 'input':
            class_attrs[field_name] = dspy.InputField(desc = description)
        else:
            class_attrs[field_name] = dspy.OutputField(desc = description)
        class_attrs["__annotations__"][field_name] = field_type

    Scores = TypedDict("Scores", {name: int for name in proposed_features}, total = True)

    class_attrs['interpretability_scores'] = dspy.OutputField(desc = "Dictionary mapping feature name to 0-10 interpretability score computed via the rubric.", ge = 0, le = 10)
    class_attrs['__annotations__']['interpretability_scores'] = Scores

    DynamicSignature = type(class_name, (dspy.Signature, ), class_attrs)
    return DynamicSignature


class _InterpretabilitiySignature(dspy.Signature):
    """Evaluate how interpretable the provided features are to a non-technical audience. 
        
        Scoring rubrics:
        Penalize per-feature interpretability score if:
            - the feature is not descriptive enough or has an unclear name (jargon / acronym without context)
            - the feature is not easily explainable to a non-technical audience.

        Penalize per-feature score SEVERELY (assign a score of 0) if a feature is leaking the label:
            - the feature is a direct leakage of the label
            - extracting the feature by an LLM is equivalent to directly predicting the label.
            - if there is a leakage, clearly mention it in the feedback

        Follow the instructions below carefully to generate the best possible feedback:
            - Provide specific, actionable feedback on the interpretability of the feature set. This could include suggestions for modifying existing features or proposing new features that could enhance model performance. You could also do nothing if the features are already good enough.
    """
    dataset_name: str = dspy.InputField(desc = "The prompt describing the task and high-level description of features to extract from texts.")
    description: str = dspy.InputField(desc = "A detailed description of the dataset, including its domain, context, and any relevant background information.")
    target_variable: str = dspy.InputField(desc = "The name of the target variable we want to predict using the extracted features.")
    example: Optional[str] = dspy.InputField(desc = "An optional example text from the dataset.")
    features: List[Feature] = dspy.InputField(desc = "List of features to be extracted from each sample in the dataset.")

    # ... added here the per-feature interpretability scores ...

    feedback: str = dspy.OutputField(desc = "A textual feedback on the feature set. If there is nothing to improve, state that explicitly.")


class InterpretabilityScorerModule(dspy.Module):

    def __init__(self, args, dataset_name, task_description, target_variable):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.target_variable = target_variable
        self.task_description = task_description

    def forward(self, features, text = None):
        generator = dspy.Predict(create_interpretability_signature_class([feature.name for feature in features]))

        result = generator(
            dataset_name = self.dataset_name,
            target_variable = self.target_variable,
            description = self.task_description,
            example = text,
            features = features,
        )

        prediction = dspy.Prediction(feedback = result.feedback, scores = result.interpretability_scores)
        return prediction


########################################################################
########################################################################
########################################################################
