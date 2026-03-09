import dspy
from typing import List, Optional
from .utils import Feature


class FeatureProposalSignature(dspy.Signature):
    """Your task is to propose a set of features to be extracted from text samples in order to predict a target variable.
    
    Instructions:
        - Analyze the provided metadata and demos to determine the domain and context of the dataset.
        - Identify the key characteristics of the dataset relevant to predicting the target variable.
        - For each identified feature, provide a clear name in snake_case, description, a list of possible values, and a specific LLM extraction query.

    Constraints:
        - Ensure features are distinct and non-redundant.
        - Note that the target variable is not explicitly present in the input text.
        - Prioritize domain-specific insights over generic ones.
        - The extraction queries must be specific and detailed to ensure high-quality feature generation.
        - Propose between 5 and 10 features.
    """

    dataset_name: str = dspy.InputField(desc = "The prompt describing the task and high-level description of features to extract from texts.")
    target_variable: str = dspy.InputField(desc = "The name of the target variable we want to predict using the extracted features.")
    description: str = dspy.InputField(desc = "A detailed description of the dataset, including its domain, context, and any relevant background information.")
    texts: Optional[List[str]] = dspy.InputField(desc = "A list of text samples from the dataset to help understand the context.")
    features: List[Feature] = dspy.OutputField(desc = "List of features to be extracted from each sample in the dataset.")


class FeatureProposalModule(dspy.Module):

    def __init__(self, args, dataset_name, task_description, target_variable):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.target_variable = target_variable
        self.task_description = task_description

        # self.generator = dspy.ChainOfThought(FeatureProposalSignature)
        self.generator = dspy.Predict(FeatureProposalSignature)

    def forward(self, text = None, label = None, texts = None) -> dspy.Prediction:
        result = self.generator(
            dataset_name = self.dataset_name,
            target_variable = self.target_variable,
            description = self.task_description,
            texts = texts,
        )
        prediction = dspy.Prediction(features = result.features, reasoning = result.reasoning if hasattr(result, "reasoning") else None)
        return prediction
