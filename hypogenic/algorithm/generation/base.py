from abc import ABC, abstractmethod
import math
import os
from typing import List

from .utils import extract_hypotheses
from ..summary_information import SummaryInformation
from ..inference import Inference
from ...tasks import BaseTask
from ...prompt import BasePrompt


class Generation(ABC):
    """Generation class"""

    def __init__(
        self,
        api,
        prompt_class: BasePrompt,
        inference_class: Inference,
        task: BaseTask,
    ):
        """Initialize the update class

        Parameters:
            api: The LLM API to call for intialization and batched hypothesis generation
                It could also be a local LLM.
            prompt_class: the class containing specific prompts for the task
            inference_class: The Inference Class to call when checking for accuracy

        """
        super().__init__()
        self.api = api
        self.prompt_class = prompt_class
        self.inference_class = inference_class
        self.task = task
        self.train_data = self.inference_class.train_data

    @abstractmethod
    def batched_initialize_hypotheses(
        self,
        num_init,
        init_batch_size,
        init_hypotheses_per_batch,
        cache_seed=None,
        max_concurrent=3,
        **kwargs
    ):
        """Initialization method for generating hypotheses. Make sure to only loop till args.num_init

        Parameters:
            args: the parsed arguments

        Returns:
            hypotheses_bank: A  dictionary with keys as hypotheses and the values as the Summary Information class
        """
        pass

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # BATCHED HYPOTHESIS LIST GENERATION                                       #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def batched_hyp_list_generation(
        self,
        example_indices: List[int],
        num_hypotheses_generate: int,
        cache_seed=None,
        **generate_kwargs
    ) -> List[str]:
        """Batched hypothesis generation method. Takes multiple examples and creates a hypothesis with them.

        Parameters:
            example_indices: the indices of examples being used to generate hypotheses
            num_hypotheses_generate: the number of hypotheses that we expect our response to generate
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number

        Returns:
            hypotheses_list: A list containing all newly generated hypotheses.
        """

        # ----------------------------------------------------------------------
        # Gather the examples to use for generation
        # ----------------------------------------------------------------------
        # Gather examples based on example_indices
        # TODO: need copy()?
        example_bank = (
            self.train_data.loc[list(example_indices)].copy().reset_index(drop=True)
        )

        # ----------------------------------------------------------------------
        # Prompt LLM to generate hypotheses
        # ----------------------------------------------------------------------
        # Batch generate a bunch of prompts based on yaml file
        prompt_input = self.prompt_class.batched_generation(
            example_bank, num_hypotheses_generate
        )

        # Batch generate responses based on the prompts that we just generated
        response = self.api.generate(
            prompt_input, cache_seed=cache_seed, **generate_kwargs
        )

        return extract_hypotheses(response, num_hypotheses_generate)

    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    # MAKE HYPOTHESES BANK                                                     #
    # ------------------------------------------------------------------------ #
    #                                                                          #
    # ------------------------------------------------------------------------ #
    def make_hypotheses_bank(
        self,
        example_indices,
        current_sample,
        alpha,
        hypotheses_list: List[str],
        cache_seed=None,
        max_concurrent=3,
        **generate_kwargs
    ):
        """
        Based on hypotheses generated by the LM, create new hypotheses_bank.

        Parameters:
            example_indices: the indices of examples being used to generate hypotheses
            current_sample: the current sample in data which the algorithm is on
            num_hypotheses_generate: the number of hypotheses that we expect our repsonse to generate
            hypotheses: a list of hypotheses generated by the LM
            alpha: eploration constant in hypogenic reward funciton
            cache_seed: If `None`, will not use cache, otherwise will use cache with corresponding seed number
            max_concurrent: the maximum number of concurrent requests to make to the API

        Returns:
            new_generated_hypotheses: A dictionary containing all newly generated hypotheses. The keys are the hypotheses and the values are the Summary Information class
        """
        idx_hyp_pair = []
        new_generated_hypotheses = {}

        for hyp in hypotheses_list:
            new_generated_hypotheses[hyp] = SummaryInformation(
                hypothesis=hyp, acc=0, num_visits=0, reward=0, correct_examples=[]
            )

            for index in example_indices:
                idx_hyp_pair.append((index, {hyp: new_generated_hypotheses[hyp]}))

        # ----------------------------------------------------------------------
        # We try to predict the ground truth labels
        # ----------------------------------------------------------------------
        preds, labels = self.inference_class.batched_predict(
            self.train_data,
            idx_hyp_pair,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs
        )
        preds, labels = preds[::-1], labels[::-1]

        # ----------------------------------------------------------------------
        # Finding the accuracy and the correct examples for each hypothesis
        # ----------------------------------------------------------------------
        for hyp in hypotheses_list:
            correct = 0
            ex = []

            # Finding accuracy
            for index in example_indices:
                prediction, actual_label = preds.pop(-1), labels.pop(-1)
                if prediction == actual_label:
                    correct += 1
                    ex.append((index, actual_label))

            # Record the accuracy, number of visits, reward, and correct examples
            acc = correct / len(example_indices)
            new_generated_hypotheses[hyp].set_accuracy(acc)
            new_generated_hypotheses[hyp].set_num_visits(len(example_indices))

            # hypogenic reward
            reward = acc + alpha * math.sqrt(
                math.log(current_sample) / len(example_indices)
            )

            new_generated_hypotheses[hyp].set_reward(reward)
            new_generated_hypotheses[hyp].set_example(ex)

        return new_generated_hypotheses
