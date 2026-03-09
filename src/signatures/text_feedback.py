import dspy
from typing import List
from .utils import Feature


class TextFeedbackSignature(dspy.Signature):
    """
        Provide feedback on the feature set based on model performance. The feedback should focus on suggesting specific changes to existing features that could enhance model accuracy or feature interpretability.
    """
    dataset_name: str = dspy.InputField(desc = "The prompt describing the task and high-level description of features to extract from texts.")
    description: str = dspy.InputField(desc = "A detailed description of the dataset, including its domain, context, and any relevant background information.")
    target_variable: str = dspy.InputField(desc = "The name of the target variable we want to predict using the extracted features.")
    features: List[Feature] = dspy.InputField(desc = "List of features to be extracted from each sample in the dataset.")
    performance_information: str = dspy.InputField(desc = "A detailed report on the model performance, including accuracy and feature importances.")

    feedback: str = dspy.OutputField(
        desc =
        "Based on the model performance and feature importances, provide feedback on the feature set. Suggest specific changes to existing features that could enhance model accuracy or feature interpretability. Be concise and focus on actionable recommendations."
    )


class TextFeedbackModule(dspy.Module):

    def __init__(self, args, dataset_name, task_description, target_variable):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.target_variable = target_variable
        self.task_description = task_description
        self.feedback_generator = dspy.ChainOfThought(TextFeedbackSignature)

    def forward(self, model_performance, features) -> dspy.Prediction:
        result = self.feedback_generator(
            dataset_name = self.dataset_name,
            description = self.task_description,
            target_variable = self.target_variable,
            features = features,
            performance_information = model_performance,
        )

        prediction = dspy.Prediction(
            feedback = result.feedback,
            reasoning = result.reasoning if hasattr(result, "reasoning") else None,
        )

        return prediction
