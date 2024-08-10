import argparse
import re
import time
import pickle
import sys
import os
import math
import json

import random
from typing import Callable, Tuple, Union
import torch
import numpy as np

from hypogenic.examples.extract_label import extract_label_register

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.data_loader import get_data
from hypogenic.utils import set_seed
from hypogenic.LLM_wrapper import LocalModelWrapper, LocalVllmWrapper
from hypogenic.algorithm.summary_information import (
    dict_to_summary_information,
)

from hypogenic.algorithm.generation import generation_register
from hypogenic.algorithm.inference import inference_register
from hypogenic.algorithm.replace import DefaultReplace
from hypogenic.algorithm.update import update_register


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_config_path", type=str, default="./data/retweet/config.yaml"
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/net/scratch/llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--max_num_hypotheses", type=int, default=20)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--old_hypothesis_file", type=str, default=None)
    parser.add_argument("--num_init", type=int, default=10)
    parser.add_argument("--num_train", type=int, default=75)
    parser.add_argument("--num_test", type=int, default=25)
    parser.add_argument("--num_val", type=int, default=25)

    parser.add_argument("--seed", type=int, default=49)

    parser.add_argument(
        "--file_name_template",
        type=str,
        default="hypotheses_training_sample_${sample}_seed_${seed}_epoch_${epoch}.json",
    )
    parser.add_argument("--sample_num_to_restart_from", type=int, default=-1)
    parser.add_argument("--epoch_to_start_from", type=int, default=0)
    parser.add_argument("--num_wrong_scale", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=5e-1)
    parser.add_argument("--update_batch_size", type=int, default=10)
    parser.add_argument("--num_hypotheses_to_update", type=int, default=1)
    parser.add_argument("--update_hypotheses_per_batch", type=int, default=5)
    parser.add_argument("--only_best_hypothesis", action="store_true", default=False)
    parser.add_argument("--save_every_n_examples", type=int, default=10)

    parser.add_argument("--init_batch_size", type=int, default=10)
    parser.add_argument("--init_hypotheses_per_batch", type=int, default=10)
    parser.add_argument("--use_cache", type=int, default=1)

    parser.add_argument("--generation_style", type=str, default="default")
    parser.add_argument("--inference_style", type=str, default="default")
    parser.add_argument("--update_style", type=str, default="default")

    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = (
            f"./outputs/retweet/{args.model_name}/hyp_{args.max_num_hypotheses}/"
        )

    return args


def main():
    # set up tools
    start_time = time.time()
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    api = LocalVllmWrapper(args.model_name, args.model_path)

    task = BaseTask(args.task_config_path, from_register=extract_label_register)

    set_seed(args.seed)
    train_data, _, _ = task.get_data(
        args.num_train, args.num_test, args.num_val, args.seed
    )
    prompt_class = BasePrompt(task)
    inference_class = inference_register.build(args.inference_style)(
        api, prompt_class, train_data, task
    )
    generation_class = generation_register.build(args.generation_style)(
        api, prompt_class, inference_class, task
    )

    update_class = update_register.build(args.update_style)(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(args.max_num_hypotheses),
        save_path=args.output_folder,
        file_name_template=args.file_name_template,
        sample_num_to_restart_from=args.sample_num_to_restart_from,
        num_init=args.num_init,
        epoch_to_start_from=args.epoch_to_start_from,
        num_wrong_scale=args.num_wrong_scale,
        k=args.k,
        alpha=args.alpha,
        update_batch_size=args.update_batch_size,
        num_hypotheses_to_update=args.num_hypotheses_to_update,
        update_hypotheses_per_batch=args.update_hypotheses_per_batch,
        only_best_hypothesis=args.only_best_hypothesis,
        save_every_n_examples=args.save_every_n_examples,
    )

    hypotheses_bank = {}
    if args.old_hypothesis_file is None:
        hypotheses_bank = update_class.batched_initialize_hypotheses(
            num_init=args.num_init,
            init_batch_size=args.init_batch_size,
            init_hypotheses_per_batch=args.init_hypotheses_per_batch,
            use_cache=args.use_cache,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample=args.num_init,
            seed=args.seed,
            epoch=0,
        )
    else:
        dict = load_dict(args.old_hypothesis_file)
        for hypothesis in dict:
            hypotheses_bank[hypothesis] = dict_to_summary_information(dict[hypothesis])
    for epoch in range(1):
        hypotheses_bank = update_class.update(
            current_epoch=epoch,
            hypotheses_bank=hypotheses_bank,
            current_seed=args.seed,
            use_cache=args.use_cache,
        )
        update_class.save_to_json(
            hypotheses_bank,
            sample="final",
            seed=args.seed,
            epoch=epoch,
        )

    # print experiment info
    print(f"Total time: {time.time() - start_time} seconds")
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     print(f'Estimated cost: {api.api.session_total_cost()}')


if __name__ == "__main__":
    main()
