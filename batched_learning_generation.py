# generate some hypothesis based on a batch of examples (in a low data regime)

import re
import time
import random
import pickle
import sys
import math
import argparse
import os
import sys
from typing import Union

from hypothesis_generation.tasks import BaseTask
from hypothesis_generation.utils import (
    LLMWrapper,
    get_num_examples,
    create_directory,
    VALID_MODELS,
    GPT_MODELS,
    set_seed,
)
from hypothesis_generation.prompt import BasePrompt
from hypothesis_generation.data_loader import get_data


def main():
    start_time = time.time()

    num_train = 25
    num_test = 0
    num_val = 0
    seed = 49
    task_config_path = "./data/retweet/config.yaml"
    model = "gpt-4o-mini"
    model_path = ""
    num_hypothesis = 5
    use_cache = 0
    hypothesis_file = f"./outputs/retweet/batched_gen_{model}_train_{num_train}_seed_{seed}_hypothesis_{num_hypothesis}.txt"

    def task_extract_label(text: Union[str, None]) -> str:
        """
        `text` follows the format "the <label> tweet got more retweets"
        """
        if text is None:
            return "other"
        text = text.lower()
        pattern = r"answer: the (\w+) tweet"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return "other"

    set_seed(seed)
    print("Getting data ...")
    task = BaseTask(task_extract_label, task_config_path)
    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
    print("Initialize LLM api ...")
    api = LLMWrapper.from_model(model, path_name=model_path, use_cache=use_cache)
    prompt_class = BasePrompt(task)
    prompt_input = prompt_class.batched_generation(train_data, num_hypothesis)
    print("Prompt: ")
    print(prompt_input)
    response = api.generate(prompt_input)
    print("prompt length: ", len(prompt_input))
    print("Response: ")
    print(response)
    with open(hypothesis_file, "w") as f:
        f.write(response)
    print("response length: ", len(response))
    print("************************************************")

    print(f"Time: {time.time() - start_time} seconds")
    # if model in GPT_MODELS:
    #     print(f"Estimated cost: {api.api.costs}")


if __name__ == "__main__":
    main()