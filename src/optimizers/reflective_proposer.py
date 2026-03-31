import json

import dspy
import tqdm
import random
from dspy.propose.utils import strip_prefix, get_dspy_source_code
from dspy.teleprompt.utils import get_signature, get_prompt_model, set_signature
from dspy.propose.grounded_proposer import (TIPS, DescribeModule,
                                            DescribeProgram)
from dspy.propose.dataset_summary_generator import create_dataset_summary

YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


class GenerateInstructionSignature(dspy.Signature):
    ("""Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task. Change the CURRENT INSTRUCTION based on the FEEDBACK provided about the PREDICTED FEATURES. Use the TIP to help you come up with a better instruction.""")
    dataset_description = dspy.InputField(desc = "A description of the dataset that we are using.", prefix = "DATASET SUMMARY:")
    program_code = dspy.InputField(format = str, desc = "Language model program designed to solve a particular task.", prefix = "PROGRAM CODE:")
    program_description = dspy.InputField(desc = "Summary of the task the program is designed to solve, and how it goes about solving it.", prefix = "PROGRAM DESCRIPTION:")
    module = dspy.InputField(desc = "The module to create an instruction for.", prefix = "MODULE:")
    module_description = dspy.InputField(desc = "Description of the module to create an instruction for.", prefix = "MODULE DESCRIPTION:")
    basic_instruction = dspy.InputField(format = str, desc = "Current instruction that need to be updated.", prefix = "CURRENT INSTRUCTION:")

    example_prediction = dspy.InputField(desc = "Example predicted features by this module.", prefix = 'PREDICTED FEATURES:')
    prediction_feedback = dspy.InputField(desc = 'Feedback on the quality of the estimated features.', prefix = 'FEEDBACK:')

    tip = dspy.InputField(format = str, desc = "A suggestion for how to go about generating the new instruction.", prefix = "TIP:")

    proposed_instruction = dspy.OutputField(desc = "Propose an instruction that will be used to prompt a Language Model to perform this task.", prefix = "PROPOSED INSTRUCTION:")


class ReflectiveProposer():

    def __init__(self, args, program, trainset, view_data_batch_size, num_demos_in_context, verbose = True):
        super().__init__()
        self.args = args
        self.program = program
        self.trainset = trainset
        self.view_data_batch_size = view_data_batch_size
        self.num_demos_in_context = num_demos_in_context
        self.verbose = verbose

        self.describe_program = dspy.Predict(DescribeProgram)
        self.describe_module = dspy.Predict(DescribeModule)
        self.instruction_proposer = dspy.Predict(GenerateInstructionSignature)

        self.data_summary = create_dataset_summary(
            trainset = trainset,
            view_data_batch_size = view_data_batch_size,
            prompt_model = get_prompt_model(None),
        )

        if self.verbose:
            print(f"DATA SUMMARY: {self.data_summary}")

        self.program_code_string = get_dspy_source_code(program)
        if self.verbose:
            print("SOURCE CODE:", self.program_code_string)

        task_demos = "No task demos provided."
        self.program_description = strip_prefix(self.describe_program(
            program_code = self.program_code_string,
            program_example = task_demos,
        ).program_description)

        if self.verbose:
            print(f"PROGRAM DESCRIPTION: {self.program_description}")

        inputs = []
        outputs = []
        for field_name, field in get_signature(self.program.predictors()[0]).fields.items():
            dspy_field_type = field.json_schema_extra.get("__dspy_field_type")
            if dspy_field_type == "input":
                inputs.append(field_name)
            else:
                outputs.append(field_name)

        module_code = f"{self.program.predictors()[0].__class__.__name__}({', '.join(inputs)}) -> {', '.join(outputs)}"

        self.module_description = self.describe_module(
            program_code = self.program_code_string,
            program_description = self.program_description,
            program_example = task_demos,
            module = module_code,
            max_depth = 10,
        ).module_description

        if self.verbose:
            print(f"MODULE DESCRIPTION: {self.module_description}")

        print(dspy.inspect_history(n = 10))

    def propose_instructions(self, demo_candidates, valset, metric, existing_instruction_candidates = None, N = -1):
        n_iters = self.args.reflection_iter 

        if existing_instruction_candidates is not None:
            instruction_candidates = existing_instruction_candidates
        else:
            instruction_candidates = {0: []}


        basic_instruction = get_signature(self.program.predictors()[0]).instructions

        for i, demos in enumerate(tqdm.tqdm(demo_candidates[0])):
            current_instruction = basic_instruction
            for iter_idx in range(n_iters):
                updated_signature = get_signature(self.program.predictors()[0]).with_instructions(current_instruction)
                set_signature(self.program.predictors()[0], updated_signature)

                while True:
                    try:
                        pred_features = [self.program(texts = [f"{example['text']}\nLabel: {example['label']}" for example in demos])]

                        if len(pred_features[0]['features']) == 0:
                            raise Exception("No features predicted!")

                        _ = metric(valset, pred_features)
                    except Exception as e:
                        print(f"Error during prediction/metric evaluation: {e}. Retrying...")
                        continue
                    break

                feature_str = {f.name: f.type for f in pred_features[0]['features'] if f.type != 'Literal' or (f.type == 'Literal' and f.literal_values)}
                feature_str = "\n".join([f"- {name}: {type_}" for name, type_ in feature_str.items()])

                selected_tip_key = random.choice(list(TIPS.keys()))
                selected_tip = TIPS[selected_tip_key]

                proposed_instruction = self.instruction_proposer(
                    dataset_description = self.data_summary,
                    program_code = self.program_code_string,
                    program_description = self.program_description,
                    module = self.program.predictors()[0].__class__.__name__,
                    module_description = self.module_description,
                    basic_instruction = current_instruction,
                    example_prediction = feature_str,
                    tip = selected_tip,
                    prediction_feedback = metric.last_value['combined_feedback'],
                ).proposed_instruction

                current_instruction = proposed_instruction
                print(f"[ReflectiveProposer [iter {iter_idx}/{n_iters}] {i}/{len(demo_candidates[0])}] Instruction:{BLUE}{proposed_instruction}{ENDC}")

            instruction_candidates[0].append(current_instruction)
            
            with open(f'checkpoints/{self.args.experiment_name}/instruction_candidates.json', 'w') as f:
                json.dump(instruction_candidates, f, indent = 4)

        return instruction_candidates