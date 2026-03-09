import os
import dspy
import json
import numpy as np
import config
import pprint
import random
import datasets
from utils import balanced_sample
from arg_utils import get_args
from factories import DATASETS, EVALUATORS, OPTIMIZERS
from signatures import FeatureProposalModule

args = get_args()
args.experiment_name = f"{args.group}:{args.name}"
pprint.pprint(vars(args))

config.ENDPOINT[args.llm_provider] = config.ENDPOINT[args.llm_provider].format(port = args.port)
args.ENDPOINT = config.ENDPOINT[args.llm_provider]

args.sampler_checkpoint_path = f'./checkpoints/samplers/{args.proposer_llm.replace("/", "-")}:{args.evaluator.replace("-feedback", "")}:{args.dataset.replace("/", "-")}:{args.max_examples_per_bag}:{args.num_examples_per_class_train}:{args.bags}:{args.temperature}:{args.seed}:{args.eps}:{args.alpha}:{args.K}/'

if args.seed != -1:
    random.seed(args.seed)
    np.random.seed(args.seed)

proposer_lm = dspy.LM(
    f'{args.llm_provider}/{args.proposer_llm}',
    api_base = config.ENDPOINT[args.llm_provider],
    api_key = 'o-parola',
    temperature = args.temperature,
    repetition_penalty = 1.2,
    top_p = 0.95,
    max_completion_tokens = 6400,
    max_tokens = 16000,
    # timeout = 7, # seconds
)

dspy.settings.configure(
    lm = proposer_lm,
    provide_traceback = True,
    track_usage = True,
    verbose = True,
    # adapter = dspy.JSONAdapter() if 'meta-llama' not in args.proposer_llm else None,
)

dspy.configure_cache(
    enable_disk_cache = args.use_cache,
    enable_memory_cache = args.use_cache,
)

# Load Dataset, split into train and validation
dataset = datasets.load_dataset(args.dataset, DATASETS[args.dataset]['subset'], split = 'train', token = config.HF_TOKEN)

if DATASETS[args.dataset]['validation_key'] is None:
    dataset = dataset.train_test_split(test_size = args.validation_size, seed = 42)
    validation_set = dataset['test']
    dataset = dataset['train']
else:
    validation_set = datasets.load_dataset(args.dataset, DATASETS[args.dataset]['subset'], split = DATASETS[args.dataset]['validation_key'], token = config.HF_TOKEN)

if 'toxic-chat' in args.dataset:
    dataset = dataset.filter(lambda x: x['jailbreaking'] == 0)
    validation_set = validation_set.filter(lambda x: x['jailbreaking'] == 0)
    # this is input-output, but let's just use the Input only, as per their paper!
    # dataset = dataset.map(lambda x: {'text': "User input: " + x['user_input'] + "\n" + "Model output: " + x['model_output']})

    # Input only
    dataset = dataset.map(lambda x: {'text': x['user_input']})
    validation_set = validation_set.map(lambda x: {'text': x['user_input']})

if 'text' not in dataset.column_names:
    dataset = dataset.rename_column(DATASETS[args.dataset]['text_column'], "text")
if 'label' not in dataset.column_names:
    dataset = dataset.rename_column(DATASETS[args.dataset]['label_column'], "label")

if 'text' not in validation_set.column_names:
    validation_set = validation_set.rename_column(DATASETS[args.dataset]['text_column'], "text")
if 'label' not in validation_set.column_names:
    validation_set = validation_set.rename_column(DATASETS[args.dataset]['label_column'], "label")

num_classes = len(set(dataset['label']))
example_trainset, annotation_set = dataset.train_test_split(test_size = 0.75, seed = 42).values()

example_trainset = balanced_sample(example_trainset, num_samples_per_class = args.num_examples_per_class_train, seed = 42)
annotation_set = balanced_sample(annotation_set, num_samples_per_class = args.annotation_set_size // num_classes, seed = 42)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

my_program = FeatureProposalModule(
    args,
    dataset_name = args.dataset,
    target_variable = DATASETS[args.dataset]['target_variable'],
    task_description = DATASETS[args.dataset]['description'],
)

mirpo_params = {
    'log_dir': f'./checkpoints-mipro/{args.experiment_name}/logs/',
    'max_errors': 1000,
    'num_trials': args.n_iters,
    'num_candidates': args.bags,
    'init_temperature': args.temperature,
    'num_threads': args.num_threads,
    'view_data_batchsize': args.bags,
    # 'max_bootstrapped_demos': args.max_examples_per_bag,
    # 'max_labeled_demos': args.max_examples_per_bag,
}

metric = EVALUATORS[args.evaluator](args = args)
optimizer = OPTIMIZERS[args.optimizer](args = args, metric = metric, mirpo_params = mirpo_params)

os.makedirs(f'checkpoints-mipro/{args.experiment_name}/', exist_ok = True)
with open(f'checkpoints-mipro/{args.experiment_name}/args.json', 'w') as f:
    json.dump(vars(args), f, indent = 4)

optimized_program = optimizer.compile(
    program = my_program,
    trainset = example_trainset,
    annotation_set = annotation_set,
    valset = validation_set,
    train_sampler = False,
    mirpo_params = mirpo_params,
)

# if type(optimized_program) == tuple:  # stupid way to check if logs are returned
#     optimized_program, logs, sampler_history = optimized_program

# print("::: Saving Optimized Program")
# optimized_program.save(f'checkpoints/{args.experiment_name}/optimized_program.json')

# with open(f'checkpoints/{args.experiment_name}/logs.json', 'w') as f:
#     logs = [{
#         'score': l['score'],
#         'features': [f.model_dump(mode = 'json') for f in l['features'].features],
#         'extended_scores': l['extended_scores'],
#     } for l in logs]
#     json.dump(logs, f, indent = 4)
