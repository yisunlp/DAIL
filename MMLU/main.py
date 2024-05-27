import argparse
from Inferencers import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import logging

logging.getLogger("transformers").setLevel(logging.CRITICAL)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, help="path to the pretrained model.")
    parser.add_argument("--method", type=str, default="DAIL")
    # Hyper parameters for DAIL
    parser.add_argument("--select_strategy", type=str, default="dpp")
    parser.add_argument("--exit_strategy", type=str, default="diverse")
    parser.add_argument("--M", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.1)
    # Hyper parameters for dpp
    parser.add_argument("--dpp_candidates", type=int, default=10)
    parser.add_argument("--scale_factor", type=float, default=0.1)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    subsets = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
               "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
               "college_medicine", "college_physics", "computer_security", "conceptual_physics",
               "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
               "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
               "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
               "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
               "high_school_physics", "high_school_psychology", "high_school_statistics",
               "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
               "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
               "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
               "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
               "professional_medicine", "professional_psychology", "public_relations", "security_studies",
               "sociology", "us_foreign_policy", "virology", "world_religions"]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 trust_remote_code=True).to(device).eval()
    if args.method == "SelfICL":
        inferencer = SelfICL(args, tokenizer, model, subsets, device)
        results = inferencer.run()
    elif args.method == "DAIL":
        sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
        inferencer = DAIL(args, tokenizer, model, sentence_model, subsets, device)
        results = inferencer.run()
    elif args.method == "zeroshot":
        inferencer = ZeroShot(args, tokenizer, model, subsets, device)
        results = inferencer.run()
    elif args.method == "fewshot":
        inferencer = FewShot(args, tokenizer, model, subsets, device)
        results = inferencer.run()
    else:
        print("Method is invalid.")
        results = None
    results = {"model": args.model, "method": args.method, "select_strategy": args.select_strategy, "M": args.M,
               "exit_strategy": args.exit_strategy, "alpha": args.alpha, "dpp_candidates": args.dpp_candidates,
               "scale_factor": args.scale_factor, "results": results}
    print("-------------------------final-results-----------------------")
    print(results)