import dspy
import random
import json

from factories import DATASETS
from dspy.teleprompt.teleprompt import Teleprompter
from optimizers.utils.miprov2_1 import MIPROv2
from sklearn.model_selection import train_test_split
from evaluators import PerBagCorrectnessScorer
import logging

# set all logging to info
logging.basicConfig(level=logging.INFO)

YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default

class PerExampleMIPROOptimizer(Teleprompter):

    def __init__(self, args, metric, mirpo_params = {}, **kwargs):
        self.args = args
        self.metric = metric
        self.optimizer = MIPROv2(args = args, metric = self.metric, **mirpo_params)
        self.proper_metric = PerBagCorrectnessScorer(self.args)

    def compile(self, program, trainset, annotation_set, valset, mirpo_params = {}, **kwargs):

        example_trainset = [dspy.Example(text = example['text'], label = DATASETS[self.args.dataset]['class_map'][example['label']]).with_inputs('text', 'label') for example in trainset]
        example_annotation_set = [dspy.Example(text = example['text'], label = DATASETS[self.args.dataset]['class_map'][example['label']]).with_inputs('text', 'label') for example in annotation_set]
        _, ann_val = train_test_split(example_annotation_set, random_state = 42, stratify = [example['label'] for example in annotation_set], test_size = 0.2)

        # select randomly 16 bag examples 
        bag_examples = random.sample(example_trainset, self.args.max_examples_per_bag)
        print(f"Selected {len(bag_examples)} examples for bag optimization.")
        print("Total length of texts:", sum([len(example.text) for example in bag_examples]))

        initial_extended_scores = None
        final_extended_scores = None
        
        try:
            pred_features = [program(texts = [f"{example['text']}\nLabel: {example['label']}" for example in bag_examples])]  # one to many
            print("predicted features:", pred_features)
            score = self.proper_metric(example_annotation_set, pred_features)
            initial_extended_scores = self.proper_metric.last_value
            print(f"Initial evaluation on annotation set: {initial_extended_scores}")
        except Exception as e:
            print(f"{YELLOW} Warning: Exception during prediction: {e}{ENDC}")
            dspy.inspect_history(n = 10)
            exit(-1)

        optimized_program = self.optimizer.compile(
            program,
            trainset = example_trainset,
            valset = ann_val,
            minibatch_size = len(ann_val),
            num_trials = self.args.n_iters,
            # **mirpo_params,
        )

        try:
            pred_features = [optimized_program()]  # one to many, but not inputs because optimization adds demo examples!
            score = self.proper_metric(example_annotation_set, pred_features)
            final_extended_scores = self.proper_metric.last_value
        except Exception as e:
            print(f"{YELLOW} Warning: Exception during prediction: {e}{ENDC}")

        print(f"Final evaluation on annotation set: {final_extended_scores}")

        with open(f'checkpoints-mipro/{self.args.experiment_name}/scores.json', 'w') as f:
            json.dump({
                'initial': initial_extended_scores,
                'final': final_extended_scores
            }, f, indent = 2)
        
        optimized_program.save(f'checkpoints-mipro/{self.args.experiment_name}/optimized_program.json')

        return optimized_program
