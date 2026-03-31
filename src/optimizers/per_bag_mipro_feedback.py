from .per_bag_mipro import ENDC, GREEN, PerBagMIPROOptimizer
from .reflective_proposer import ReflectiveProposer


class PerBagMIPROWithFeedbackOptimizer(PerBagMIPROOptimizer):

    def __init__(self, args, metric, mirpo_params = {}):
        super().__init__(args, metric, **mirpo_params)

    def _propose_instructions(self, program, trainset, valset, demo_candidates, trial_logs, N, **kwargs):
        # Use the reflective instruction proposer that receives text feedback on the output.
        # It receives as input the metric and a validation set to guide the proposals.
        # Current instruction, summary of demos, the output features on this input and the metric output. Returns a proposed new instruction.
        # Afterwards we will have a list of instruction candidates per demo_candidate as before. Do optuna optimization as before.
        print("::: Proposing instruction candidates using the reflective proposer...")

        proposer = ReflectiveProposer(
            args = self.args,
            program = program,
            trainset = trainset,
            view_data_batch_size = self.args.bags,
            num_demos_in_context = self.args.max_examples_per_bag,
        )

        instruction_candidates = proposer.propose_instructions(
            demo_candidates = demo_candidates,
            valset = valset,
            metric = self.metric,
            N = N,
            **kwargs,
        )

        return instruction_candidates
