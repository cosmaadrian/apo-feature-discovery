import dspy
import numpy as np
import pandas as pd
from typing import Protocol
from .utils import train_lr_classifier
from factories import DATASETS
from collections import defaultdict
from signatures.text_feedback import TextFeedbackModule
from signatures.interpretability import InterpretabilityScorerModule
from signatures.target_predictor import PredictorModule
from signatures.feature_extractor import ExtractorModule


class Scorer(Protocol):

    def __init__(self, args):
        self.args = args
        self.parallel = dspy.Parallel(num_threads = self.args.num_threads, max_errors = 100)
        self.extractor_program = ExtractorModule(args = args)

        self.last_value = None

    def __call__(self, example, pred, trace = None):
        raise NotImplementedError


########################################################################################
########################################################################################
########################################################################################


class PerExampleCorrectnessScorer(Scorer):
    """
        This scorer extracts features for a single example, predicts the class using the prediction program, and returns 1.0 if the prediction is correct, 0.0 otherwise.
    """

    def __init__(self, args):
        super().__init__(args)
        self.prediction_program = PredictorModule(
            args = args,
            dataset_name = args.dataset,
            target_variable = DATASETS[args.dataset]['target_variable'],
            task_description = DATASETS[args.dataset]['description'],
            possible_classes = DATASETS[args.dataset]['possible_classes'],
        )

    def __call__(self, example, pred, trace = None):
        with dspy.settings.context(temperature = 0.0):
            extracted_features = self.extractor_program(text = example['text'], features = pred['features'])
            feature_spec = [f for f in pred['features'] if f.type != 'Literal' or (f.type == 'Literal' and f.literal_values)]  # filter out empty literals

        with dspy.settings.context(temperature = 0.0):
            predicted_class = self.prediction_program(feature_spec = feature_spec, feature_values = extracted_features).target_variable

        self.last_value = float(predicted_class == example['label'])
        return self.last_value


class PerExampleCorrectnessAndInterpretabilityScorer(Scorer):
    """
        This scorer computes a combined score based on correctness and interpretability for a single example.
    """

    def __init__(self, args):
        super().__init__(args)
        self.correctness_scorer = PerExampleCorrectnessScorer(args)
        self.interpretability_program = InterpretabilityScorerModule(
            args = args,
            dataset_name = args.dataset,
            target_variable = DATASETS[args.dataset]['target_variable'],
            task_description = DATASETS[args.dataset]['description'],
        )

    def __call__(self, example, pred, trace = None):
        correctness_score = self.correctness_scorer(example, pred, trace)
        features = [f for f in pred['features'] if f.type != 'Literal' or (f.type == 'Literal' and f.literal_values)]  # filter out empty literals

        with dspy.settings.context(temperature = 0.0):
            interpretability_score = self.interpretability_program(features = features)

        average_interpretability = sum(interpretability_score.scores.values()) / len(interpretability_score.scores)
        average_interpretability = average_interpretability / 10.0  # normalize to [0, 1]

        self.last_value = {
            'correctness_score': correctness_score,
            'average_interpretability': average_interpretability,
            'combined_score': correctness_score * 0.75 + average_interpretability * 0.25,
            'per_feature_interpretability': interpretability_score.scores,
        }

        return correctness_score * 0.75 + average_interpretability * 0.25


########################################################################################
########################################################################################
########################################################################################


class PerBagScorer(Scorer):

    def extract_features(self, example, preds):
        with dspy.settings.context(temperature = 0.0):
            extracted_features = self.parallel([(self.extractor_program, dspy.Example(text = _example['text'], features = preds[0].features).with_inputs('text', 'features')) for _example in example])
        return extracted_features

    def construct_feature_dataframe(self, example, pred):
        annotated_samples = self.extract_features(example, pred)

        df = defaultdict(list)
        for original_example, _example in zip(example, annotated_samples, strict = True):
            for key, value in _example.items():
                df['feat:' + key].append(value)

            df['text'].append(original_example['text'])
            df['label'].append(original_example['label'])

        return pd.DataFrame(df)


class PerBagCorrectnessScorer(PerBagScorer):
    """
        This scorer extracts features for all examples in the bag, trains a logistic regression classifier on the extracted features, and returns the accuracy on a held-out test set.
    """

    def __call__(self, example, pred, trace = None):
        df = self.construct_feature_dataframe(example, pred)
        feature_types = {f.name: f.type for f in pred[0]['features'] if f.type != 'Literal' or (f.type == 'Literal' and f.literal_values)}
        output = train_lr_classifier(df, feature_types = feature_types)

        self.last_value = output
        return output['metrics']['f1:macro']


class PerBagCorrectnessAndInterpretabilityScorer(PerBagCorrectnessScorer):
    """
        This scorer extracts features for all examples in the bag, trains a logistic regression classifier on the extracted features, and returns a combined score based on accuracy and interpretability.
    """

    def __init__(self, args):
        super().__init__(args)
        self.interpretability_program = InterpretabilityScorerModule(
            args = args,
            dataset_name = args.dataset,
            target_variable = DATASETS[args.dataset]['target_variable'],
            task_description = DATASETS[args.dataset]['description'],
        )

    def __call__(self, example, pred, trace = None):
        accuracy = super().__call__(example, pred, trace)

        with dspy.settings.context(temperature = 0.0):
            interpretability_scores = self.interpretability_program(features = pred[0]['features'])

        average_interpretability = np.mean([score for score in interpretability_scores.scores.values()]) / 10.0
        combined_score = accuracy * 0.75 + average_interpretability * 0.25

        self.last_value = {
            **self.last_value,
            **{
                'feedback': interpretability_scores.feedback,
                'per_feature_interpretability': interpretability_scores.scores,
                'average_interpretability': average_interpretability,
                'combined_score': combined_score,
            }
        }

        return combined_score


class PerBagCorrectnessAndInterpretabilityWithFeedbackScorer(PerBagCorrectnessAndInterpretabilityScorer):

    def __init__(self, args):
        super().__init__(args)

        self.feature_feedback_program = TextFeedbackModule(
            args = args,
            dataset_name = args.dataset,
            target_variable = DATASETS[args.dataset]['target_variable'],
            task_description = DATASETS[args.dataset]['description'],
        )

    def __call__(self, example, pred, trace = None):
        df = self.construct_feature_dataframe(example, pred)

        features = [f for f in pred[0]['features'] if f.type != 'Literal' or (f.type == 'Literal' and f.literal_values)]  # filter out empty literals
        feature_types = {f.name: f.type for f in pred[0]['features'] if f.type != 'Literal' or (f.type == 'Literal' and f.literal_values)}

        output = train_lr_classifier(df, feature_types)
        accuracy = output['metrics']['f1:macro']
        feature_importances = output['feature_importances']

        with dspy.settings.context(temperature = 0.0):
            interpretability_scores = self.interpretability_program(features = pred[0]['features'])

        average_interpretability = np.mean([score for score in interpretability_scores.scores.values()]) / 10.0
        combined_score = accuracy * 0.75 + average_interpretability * 0.25

        formatted_performance_results = self.format_performance_results(df, feature_types, accuracy, feature_importances)
        performance_feedback = self.feature_feedback_program(model_performance = formatted_performance_results, features = features)

        feedback = f"Interpretability feedback: {interpretability_scores.feedback}\n"
        feedback += f"Performance feedback: {performance_feedback.feedback}"

        self.last_value = {
            **output,
            **{
                'average_interpretability': average_interpretability,
                'per_feature_interpretability': interpretability_scores.scores,
                'combined_score': combined_score,
                'interpretability_feedback': interpretability_scores.feedback,
                'performance_feedback': performance_feedback.feedback,
                'combined_feedback': feedback,
            }
        }

        return combined_score

    def format_performance_results(self, df, feature_types, accuracy, feature_importances):
        dtype_format = {name: "(" + f"{(dtype if dtype != 'Literal' else 'categorical')}" + (" - {" + ', '.join(c for c in df['feat:' + name].unique()) + '}' if dtype == 'Literal' else "") + ")" for name, dtype in feature_types.items()}
        formatted_feature_importances = [f"{fi['feature_name']} {dtype_format[fi['feature_name']]} = (Shap: {fi['importance']}, Mutual Info: {fi['mi']}, Distributional Coverage (entropy): {fi['coverage']})" for fi in feature_importances]

        output = f"Accuracy: {round(accuracy * 100, 2)}%\n\n"
        output += "Feature Importances:\n"
        output += "SHAP Importance: strength of influence on model predictions.\n"
        output += 'Mutual Information: measures dependency between feature and label.\n'
        output += "Coverage Interpretation: 1.0 = uniform and symmetric (broad coverage); < 0.5 highly concentrated or skewed.\n"
        output += "\n".join([str(fi) for fi in formatted_feature_importances])

        return output


########################################################################################
########################################################################################
########################################################################################
