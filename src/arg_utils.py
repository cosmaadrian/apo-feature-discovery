import argparse


def get_args():
    parser = argparse.ArgumentParser(description = "Unsupervised interpretable feature extraction")
    parser.add_argument('--name', type = str, required = False, default = 'test', help = 'wandb run name')
    parser.add_argument('--group', type = str, required = False, default = 'test', help = 'wandb run group')
    parser.add_argument('--mode', type = str, required = False, default = 'online', help = 'wandb mode (online, offline, disabled)')
    parser.add_argument('--notes', type = str, required = False, default = 'test run', help = 'wandb run notes')
    parser.add_argument('--use_cache', action = 'store_true', help = 'Whether to use disk cache for LLM calls.')

    parser.add_argument('--port', type = int, required = False, default = 3000, help = 'Port for the LLM API endpoint.')

    parser.add_argument('--optimizer', type = str, required = False, default = 'example-mipro', help = 'Optimizer to use for feature selection.')
    parser.add_argument('--evaluator', type = str, required = False, default = 'example-correctness-interpretability', help = 'Evaluator to use for feature selection.')
    parser.add_argument('--dataset', type = str, required = False, default = 'stanfordnlp/imdb', help = 'Huggingface Dataset to use.')

    parser.add_argument('--proposer_llm', type = str, required = False, default = 'Qwen/Qwen3-14B', help = 'LLM to use for signature generation.')

    parser.add_argument('--llm_provider', type = str, required = False, default = 'hosted_vllm', help = 'LLM provider to use.')
    parser.add_argument('--reflection_iter', type = int, required = False, default = 1, help = 'Number of reflection iterations to perform per instruction.')

    parser.add_argument('--temperature', type = float, required = False, default = 1.0, help = 'Temperature to use for LLM calls.')

    parser.add_argument('--max_examples_per_bag', type = int, required = False, default = 5, help = 'Maximum number of training samples to use in as in-context examples. [4-16]')
    parser.add_argument('--num_examples_per_class_train', type = int, required = False, default = 20, help = 'Number of training examples per class to use for feature proposal.')
    parser.add_argument('--annotation_set_size', type = int, required = False, default = 16, help = 'Number of samples in each validation batch. The larger the more stable the feedback.')
    parser.add_argument('--validation_size', type = int, required = False, default = 256, help = 'Number of samples to use for validation.')
    parser.add_argument('--demo_rounds', type = int, required = False, default = 4, help = 'Number of rounds to use for demo selection.')
    parser.add_argument('--n_iters', type = int, required = False, default = 10, help = 'Number of optimization iterations to perform.')

    parser.add_argument('--bags', type = int, required = False, default = 2, help = 'Number of bags to process.')
    parser.add_argument('--num_threads', type = int, required = False, default = 4, help = 'Number of threads to use for parallel predictions.')

    parser.add_argument('--eps', type = float, required = False, default = 0.1, help = 'Exploration factor for online batch sampler.')
    parser.add_argument('--dont_train_sampler', action = 'store_true', required = False, help = 'Whether to train the online batch sampler.')

    parser.add_argument('--alpha', type = float, required = False, default = 0.25, help = 'Learning rate for online batch sampler.')
    parser.add_argument('--K', type = float, required = False, default = 1, help = 'The K for the batch sampler {1, 2}.')

    parser.add_argument('--checkpoint_dir', type = str, required = False, default = './checkpoints/', help = 'Number of threads to use for parallel predictions.')

    parser.add_argument('--seed', type = int, required = False, default = 69, help = 'Random seed.')
    parser.add_argument('--verbose', type = bool, required = False, default = True, help = 'Whether to print out progress.')
    args = parser.parse_args()
    return args
