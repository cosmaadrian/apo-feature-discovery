import os
import dspy
import numpy as np
import optuna
from factories import DATASETS
from dspy.propose import GroundedProposer
from batch_sampler import KSampler
from optuna.distributions import CategoricalDistribution
from dspy.teleprompt.utils import (get_signature, set_signature,
                                   print_full_program)
import json

from dspy.teleprompt.teleprompt import Teleprompter

YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


class PerBagMIPROOptimizer(Teleprompter):

    def __init__(self, args, metric, **kwargs):
        self.args = args
        self.metric = metric
        self.compiled = False
        self.sampler = None

    def evaluate_bags(self, program, bags, valset, bag_indices):
        bag_scores = []

        for bag_idx, bag_examples in enumerate(bags):
            print(f":::: Evaluating bag {bag_idx + 1} / {len(bags)}")
            print(":::: Indices:", sorted(bag_indices[bag_idx]))
            try:
                pred_features = [program(texts = [f"{example['text']}\nLabel: {example['label']}" for example in bag_examples])]  # one to many
            except Exception as e:
                print(f"{YELLOW} Warning: Exception during prediction: {e}{ENDC}")
                continue

            score = self.metric(valset, pred_features)

            print(f":::: Bag {bag_idx + 1} score: {score}")
            bag_scores.append((score, self.metric.last_value))

        return bag_scores

    def _train_sampler(self, program, example_trainset, example_annotation_set):
        if self.args.dont_train_sampler:
            print("::: Skipping sampler training as per configuration.")
            return []

        assert False, "Disabled for now!"
        print(f"Running sampling for {self.args.demo_rounds} rounds of {self.args.bags // 4} bags each...")
        for round_idx in range(len(self.sampler.history), self.args.demo_rounds):
            bags = []
            bag_indices = []

            for _ in range(self.args.bags // 4):  # Reduced number of bags per round for efficiency
                indices = self.sampler.sample(k = self.args.max_examples_per_bag)
                sampled_example_trainset = [example_trainset[i] for i in indices]

                bag_indices.append(indices)
                bags.append(sampled_example_trainset)

            # Score each batch and update sampler.
            print(f":::: Evaluating bag rounds ... {round_idx} / {self.args.demo_rounds}")
            bag_scores = self.evaluate_bags(program, bags, example_annotation_set, bag_indices)

            # check if bag scores are not empty of all zeroes
            if len(bag_scores) == 0 or all(score == 0.0 for score, _ in bag_scores):
                print(f"{YELLOW} Warning: All bag scores are zero or no scores were computed. Skipping sampler update for this round.{ENDC}")
                continue

            self.sampler.history.append({'round': round_idx, 'bag_indices': bag_indices, 'bag_scores': [b for b, _ in bag_scores], 'extended_scores': [ext for _, ext in bag_scores]})
            print(f"::: Average bag score this round: {np.mean([b for b, _ in bag_scores])} ± {np.std([b for b, _ in bag_scores])}")

            self.sampler.update_many(list(zip(bag_indices, [b for b, _ in bag_scores])))
            self.sampler.save()
            print(self.sampler.w)

        print("::: Sampling complete.")

        with open(f'checkpoints/{self.args.experiment_name}/sampler_history.json', 'w') as f:
            json.dump(self.sampler.history, f, indent = 4)

        return self.sampler.history

    def _propose_instructions(self, program, trainset, valset, demo_candidates, trial_logs, N):
        print("::: Proposing instruction candidates using the grounded proposer...")
        proposer = GroundedProposer(
            program = program,
            trainset = trainset,
            prompt_model = None,
            view_data_batch_size = self.args.bags,
            program_aware = True,
            use_dataset_summary = True,
            use_task_demos = False,
            num_demos_in_context = self.args.max_examples_per_bag,
            use_tip = True,
            set_tip_randomly = True,
            use_instruct_history = False,
            set_history_randomly = False,
            verbose = False,
            init_temperature = self.args.temperature,
        )

        instruction_candidates = proposer.propose_instructions_for_program(trainset = trainset, program = program, demo_candidates = demo_candidates, trial_logs = trial_logs, N = N)

        return instruction_candidates

    def compile(self, program, trainset, valset, annotation_set, train_sampler = True, **kwargs):
        if self.compiled:
            print('[👑] Program already compiled. Tuning further, but starting from the previously trained sampler.')

        example_trainset = [dspy.Example(text = example['text'], label = DATASETS[self.args.dataset]['class_map'][example['label']]).with_inputs('text', 'label') for example in trainset]
        example_annotation_set = [dspy.Example(text = example['text'], label = DATASETS[self.args.dataset]['class_map'][example['label']]).with_inputs('text', 'label') for example in annotation_set]

        # # Step 1: Train the sampler to find the best demo candidates.
        if self.sampler is None:
            self.sampler = KSampler(self.args, n_items = len(example_trainset))

        if not train_sampler:
            self.sampler = self.sampler.load(self.args.sampler_checkpoint_path)

        if self.sampler is None:
            self.sampler = KSampler(self.args, n_items = len(example_trainset))
            self._train_sampler(program, example_trainset, example_annotation_set)
        else:
            print("::: Loaded pretrained sampler.")
            if len(self.sampler.history) < self.args.demo_rounds:
                self.sampler.n_updates = len(self.sampler.history)
                print(f"::: Loaded sampler is not fully trained ({len(self.sampler.history)}/{self.args.demo_rounds}), continuing training...")
                self._train_sampler(program, example_trainset, example_annotation_set)
    
        # if exists and is not empty
        if os.path.exists(f'checkpoints/{self.args.experiment_name}/demo_idxs.json') and os.path.getsize(f'checkpoints/{self.args.experiment_name}/demo_idxs.json') > 0:
            print("::: Loading demo candidates from disk.")
            with open(f'checkpoints/{self.args.experiment_name}/demo_idxs.json', 'r') as f:
                best_demos_idxs = json.load(f)
        else:
            print("::: Identifying best demo candidates using the trained sampler...")
            best_demos_idxs = [self.sampler.sample(k = self.args.max_examples_per_bag, explore = False) for _ in range(self.args.bags)]
            scores = [self.sampler.estimate_set_score(best_demos_idx) for best_demos_idx in best_demos_idxs]

            # sort best_demo_idxs by scores descending
            best_demos_idxs = [x for _, x in sorted(zip(scores, best_demos_idxs), key = lambda pair: pair[0], reverse = False)]

            os.makedirs(f'checkpoints/{self.args.experiment_name}/', exist_ok = True)
            with open(f'checkpoints/{self.args.experiment_name}/demo_idxs.json', 'w') as f:
                json.dump(best_demos_idxs, f, indent = 4)

        print("::: Best demos indices:", best_demos_idxs)
        best_demos = [[example_trainset[i] for i in best_demos_idxs[j]] for j in range(len(best_demos_idxs))]
        best_demos = {0: best_demos}

        if os.path.exists(f'checkpoints/{self.args.experiment_name}/instruction_candidates.json') and os.path.getsize(f'checkpoints/{self.args.experiment_name}/instruction_candidates.json') > 0:
            print("::: Loading instruction candidates from disk.")
            with open(f'checkpoints/{self.args.experiment_name}/instruction_candidates.json', 'r') as f:
                instruction_candidates = json.load(f)
                instruction_candidates = {int(k): v for k, v in instruction_candidates.items()}
        else:
            # Step 2: Using the demos, propose instructions using the feature proposer.
            instruction_candidates = self._propose_instructions(program = program, trainset = example_trainset, valset = example_annotation_set, demo_candidates = best_demos, trial_logs = {}, N = len(best_demos[0]))

            with open(f'checkpoints/{self.args.experiment_name}/instruction_candidates.json', 'w') as f:
                json.dump(instruction_candidates, f, indent = 4)

        print("::: Proposed instruction candidates:")
        for i, instrs in enumerate(instruction_candidates.items()):
            for value in instrs[1]:
                print(f"Instruction:{BLUE}{value}{ENDC}")

        # Step 3: Optimize prompt parameters: use optuna to find the best combinations of instructions and demos.
        best_program = self._optimize_prompt_parameters(
            program = program,
            instruction_candidates = instruction_candidates,
            demo_candidates = best_demos,
            annotation_set = example_annotation_set,
        )

        self.compiled = True
        return best_program

    def _get_param_distributions(self, instruction_candidates, demo_candidates):
        param_distributions = {}
        for i in range(len(instruction_candidates)):
            param_distributions[f"{i}_instruction"] = CategoricalDistribution(range(len(instruction_candidates[i])))
            if demo_candidates:
                param_distributions[f"{i}_demos"] = CategoricalDistribution(range(len(demo_candidates[i])))

        return param_distributions

    def _select_and_insert_instructions_and_demos(self, candidate_program, instruction_candidates, demo_candidates, trial, trial_logs, trial_num):
        chosen_params = []
        raw_chosen_params = {}

        for i, predictor in enumerate(candidate_program.predictors()):
            # Select instruction
            instruction_idx = trial.suggest_categorical(f"{i}_instruction", range(len(instruction_candidates[i])))
            selected_instruction = instruction_candidates[i][instruction_idx]
            updated_signature = get_signature(predictor).with_instructions(selected_instruction)

            set_signature(predictor, updated_signature)

            trial_logs[trial_num][f"{i}_instruction"] = instruction_idx
            chosen_params.append(f"Predictor {i}: Instruction {instruction_idx}")
            raw_chosen_params[f"{i}_instruction"] = selected_instruction

            # Select demos if available
            if demo_candidates:
                demos_idx = trial.suggest_categorical(f"{i}_demos", range(len(demo_candidates[i])))
                predictor.demos = demo_candidates[i][demos_idx]

                trial_logs[trial_num][f"{i}_demos"] = demos_idx
                chosen_params.append(f"Predictor {i}: Few-Shot Set {demos_idx}")
                raw_chosen_params[f"{i}_demos"] = demo_candidates[i][demos_idx]

        return chosen_params, raw_chosen_params

    def _optimize_prompt_parameters(self, program, instruction_candidates, demo_candidates, annotation_set):
        optuna_study_path = f"./checkpoints/optuna_studies/{self.args.experiment_name}.log"
        num_trials = self.args.n_iters
        num_trials = min(num_trials, self.args.bags * self.args.bags) 
        default_score = 0.0
        trial_logs = {}
        trial_logs[1] = {}
        score_data = []
        best_score = -float('inf')
        best_program = None
        trials_so_far = 0

        if not (os.path.exists(f'./checkpoints/{self.args.experiment_name}/optimization_logs.json') and os.path.getsize(f'./checkpoints/{self.args.experiment_name}/optimization_logs.json') > 0):
            # make a prediction
            # Evaluate the candidate program
            the_demos = demo_candidates[0][0]

            ################################################

            try:
                pred_features = [program(texts = [f"{example['text']}\nLabel: {example['label']}" for example in the_demos])]  # one to many
                # pred_features = [program(texts = [f"{example['text']}" for example in the_demos])]  # one to many
                dspy.inspect_history(n = 3)
                score = self.metric(annotation_set, pred_features)
                default_score = score
            except Exception as e:
                print(f"{YELLOW} Warning: Exception during initial evaluation: {e}{ENDC}")
                pred_features = [None]
            ################################################
   
            best_score = default_score
            best_program = program.deepcopy()

            program.predictors()[0].demos = the_demos
            score_data = [{"score": best_score, "program": program.deepcopy(), 'features': pred_features[0], 'extended_scores': self.metric.last_value}]  # score, prog, full_eval
            program.predictors()[0].demos = []

        else:
            with open(f'./checkpoints/{self.args.experiment_name}/optimization_logs.json', 'r') as f:
                lines = f.readlines()
                log_lines = [json.loads(line) for line in lines]
                trials_so_far = len(log_lines)

                print(len(log_lines), num_trials)
                if len(log_lines) >= num_trials:
                    print()
                    print()
                    print(f"{GREEN}Optuna study already has {len(lines)} trials, which is >= {num_trials}. Skipping optimization.{ENDC}")
                    print()
                    print()
                    return program, [], self.sampler.history

        # Define the objective function
        def objective(trial):
            nonlocal program, best_program, best_score, trial_logs, score_data

            trial_num = trial.number + 1
            print(f"===== Trial {trial_num} / {num_trials} =====")

            trial_logs[trial_num] = {}

            # Create a new candidate program
            candidate_program = program.deepcopy()

            # Choose instructions and demos, insert them into the program
            _, raw_chosen_params = self._select_and_insert_instructions_and_demos(
                candidate_program,
                instruction_candidates,
                demo_candidates,
                trial,
                trial_logs,
                trial_num,
            )

            # Log assembled program
            print("Evaluating the following candidate program...\n")
            print_full_program(candidate_program)

            # Evaluate the candidate program
            the_demos = candidate_program.predictors()[0].demos
            candidate_program.predictors()[0].demos = []

            ################################################
            try:
                pred_features = [candidate_program(texts = [f"{example['text']}\nLabel: {example['label']}" for example in the_demos])]  # one to many
                # pred_features = [candidate_program(texts = [f"{example['text']}" for example in the_demos])]  # one to many
            except Exception as e:
                print(f"{YELLOW} Warning: Exception during prediction: {e}{ENDC}")
                if "litellm.APIError: APIError: Hosted_vllmException - Connection error" in str(e):
                    print("Exiting due to connection error...")
                    exit(-1)
                return 0.0
            ################################################

            try:
                score = self.metric(annotation_set, pred_features)
            except Exception as e:
                print(f"{YELLOW} Warning: Exception during evaluation: {e}{ENDC}")
                if "litellm.APIError: APIError: Hosted_vllmException - Connection error" in str(e):
                    print("Exiting due to connection error...")
                    exit(-1)
                return 0.0

            # Restore demos
            candidate_program.predictors()[0].demos = the_demos

            # Update best score and program
            if score > best_score:
                best_score = score
                best_program = candidate_program.deepcopy()

                best_program.save(f'checkpoints/{self.args.experiment_name}/best_program:{trial_num}:{best_score}.json')
                print(f"{GREEN}Best full score so far!{ENDC} Score: {score}")

            print(self.metric.last_value)
            l = {"score": score, "program": candidate_program, 'features': pred_features[0], 'extended_scores': self.metric.last_value}
            with open(f'checkpoints/{self.args.experiment_name}/optimization_logs.json', 'a') as f:
                f.write(
                    json.dumps({
                        'score': l['score'],
                        'features': [f.model_dump(mode = 'json') for f in l['features'].features],
                        'extended_scores': l['extended_scores'],
                    }) + '\n'
                )

            # Log evaluation results
            score_data.append(l)
            return score

        sampler = optuna.samplers.TPESampler(seed = self.args.seed, multivariate = True)

        os.makedirs("./checkpoints/optuna_studies/", exist_ok = True)
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(optuna_study_path))
        study = optuna.create_study(direction = "maximize", sampler = sampler, study_name = f"{self.args.experiment_name}", storage = storage, load_if_exists = True)

        default_params = {f"{i}_instruction": 0 for i in range(len(program.predictors()))}
        if demo_candidates:
            default_params.update({f"{i}_demos": 0 for i in range(len(program.predictors()))})

        # Add default run as a baseline in optuna
        trial = optuna.trial.create_trial(
            params = default_params,
            distributions = self._get_param_distributions(instruction_candidates, demo_candidates),
            value = default_score,
        )

        study.add_trial(trial)
        study.optimize(objective, n_trials = num_trials - trials_so_far)

        if best_program is not None:
            best_program.trial_logs = trial_logs
            best_program.score = best_score

        print(f"Returning best identified program with score {best_score}!")
        return best_program, score_data, self.sampler.history
