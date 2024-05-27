import argparse
from sentence_transformers import SentenceTransformer
import torch
from inferencers import DAIL
import openai
import random
import numpy as np
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-instruct")
    parser.add_argument("--method", type=str, default="DAIL")
    # Hyper parameters for DAIL
    parser.add_argument("--select_strategy", type=str, default="dpp")
    parser.add_argument("--exit_strategy", type=str, default="diverse")
    parser.add_argument("--M", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0)
    # Hyper parameters for dpp
    parser.add_argument("--dpp_candidates", type=int, default=10)
    parser.add_argument("--scale_factor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_key", type=str)
    arguments = parser.parse_args()
    return arguments


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    openai.api_key = args.api_key
    subsets = {
        'boolean_expressions': r'(True|False)',
        'causal_judgement': r'(Yes|No)',
        'date_understanding': r'[A-F]',
        'disambiguation_qa': r'[A-C]',
        'formal_fallacies': r'(invalid|valid)',
        'geometric_shapes': r'[A-K]',
        'hyperbaton': r'[A-B]',
        'logical_deduction_five_objects': r'[A-E]',
        'logical_deduction_seven_objects': r'[A-G]',
        'logical_deduction_three_objects': r'[A-C]',
        'movie_recommendation': r'[A-E]',
        'navigate': r'(Yes|No)',
        'penguins_in_a_table': r'[A-E]',
        'reasoning_about_colored_objects': r'[A-R]',
        'ruin_names': r'[A-D]',
        'salient_translation_error_detection': r'[A-F]',
        'snarks': r'[A-B]',
        'sports_understanding': r'(yes|no)',
        'temporal_sequences': r'[A-D]',
        'tracking_shuffled_objects_five_objects': r'[A-E]',
        'tracking_shuffled_objects_seven_objects': r'[A-G]',
        'tracking_shuffled_objects_three_objects': r'[A-C]',
        'web_of_lies': r'(Yes|No)'
    }

    label_space_map = {
        'boolean_expressions': ['True', 'False'],
        'causal_judgement': ['Yes', 'No'],
        'date_understanding': ['A', 'B', 'C', 'D', 'E', 'F'],
        'disambiguation_qa': ['A', 'B', 'C'],
        'formal_fallacies': ['invalid', 'valid'],
        'geometric_shapes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
        'hyperbaton': ['A', 'B'],
        'logical_deduction_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'logical_deduction_seven_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'logical_deduction_three_objects': ['A', 'B', 'C'],
        'movie_recommendation': ['A', 'B', 'C', 'D', 'E'],
        'navigate': ['Yes', 'No'],
        'penguins_in_a_table': ['A', 'B', 'C', 'D', 'E'],
        'reasoning_about_colored_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                                            'P', 'Q', 'R'],
        'ruin_names': ['A', 'B', 'C', 'D'],
        'salient_translation_error_detection': ['A', 'B', 'C', 'D', 'E', 'F'],
        'snarks': ['A', 'B'],
        'sports_understanding': ['yes', 'no'],
        'temporal_sequences': ['A', 'B', 'C', 'D'],
        'tracking_shuffled_objects_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'tracking_shuffled_objects_seven_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'tracking_shuffled_objects_three_objects': ['A', 'B', 'C'],
        'web_of_lies': ['Yes', 'No']
    }
    is_choices = {
        'date_understanding': ['A', 'B', 'C', 'D', 'E', 'F'],
        'disambiguation_qa': ['A', 'B', 'C'],
        'geometric_shapes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'],
        'hyperbaton': ['A', 'B'],
        'logical_deduction_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'logical_deduction_seven_objects': ['A', 'B', 'C', 'D', 'E'],
        'logical_deduction_three_objects': ['A', 'B', 'C', 'D', 'E'],
        'movie_recommendation': ['A', 'B', 'C', 'D', 'E'],
        'penguins_in_a_table': ['A', 'B', 'C', 'D', 'E'],
        'reasoning_about_colored_objects': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                                            'P', 'Q', 'R'],
        'ruin_names': ['A', 'B', 'C', 'D'],
        'salient_translation_error_detection': ['A', 'B', 'C', 'D', 'E', 'F'],
        'snarks': ['A', 'B'],
        'temporal_sequences': ['A', 'B', 'C', 'D'],
        'tracking_shuffled_objects_five_objects': ['A', 'B', 'C', 'D', 'E'],
        'tracking_shuffled_objects_seven_objects': ['A', 'B', 'C', 'D', 'E'],
        'tracking_shuffled_objects_three_objects': ['A', 'B', 'C', 'D', 'E']
    }
    with open("bbh/bbh_task_description.json", "r") as f:
        task_description = json.load(f)
    device = torch.device(args.device)
    sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
    inferencer = DAIL(args, sentence_model, subsets, label_space_map, task_description, is_choices, device)
    results = inferencer.run()

    results = {"model": args.model, "tokens": inferencer.tokens, "method": args.method,
               "select_strategy": args.select_strategy, "M": args.M,
               "exit_strategy": args.exit_strategy, "alpha": args.alpha, "dpp_candidates": args.dpp_candidates,
               "scale_factor": args.scale_factor, "results": results}
    print("-------------------------final-results-----------------------")
    print(results)

