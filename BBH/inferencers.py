import torch
import dataclasses
from retriever import DynamicReteiever
import openai
import tiktoken
import json
from tqdm import tqdm
import re
import random


@dataclasses.dataclass
class Sample:
    idx: int
    question: str
    label: str
    embed: torch.TensorType
    pseudo_label: str or None
    demonstration: str or None


class DAIL:
    """
    Inference code for DAIL. You can inference your data with two steps:
    1). Init:             inferencer = DAIL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, sentence_model, subsets, label_space_map, task_description, is_choices, device):
        self.args = args
        self.sentence_model = sentence_model
        self.subsets = subsets
        self.label_space_map = label_space_map
        self.task_description_map = task_description
        self.is_choices = is_choices
        self.is_choice = False
        self.pattern = None
        self.label_space = None
        self.task_description = None
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.retriever = DynamicReteiever(args)
        self.encoding = tiktoken.encoding_for_model(args.model)
        self.tokens = 0
        self.global_test_sample_num = 0
        self.global_right_sample_num = 0

    def get_embedding(self, sentence):
        embedding = self.sentence_model.encode([sentence], convert_to_tensor=True)
        return embedding

    def get_query_response(self, query):
        response = openai.Completion.create(
            model=self.args.model,
            prompt=query,
            max_tokens=10,
            temperature=0
        )
        response = response.choices[0].text.strip()
        self.tokens += len(self.encoding.encode(query))
        self.tokens += len(self.encoding.encode(response))
        matches = re.findall(self.pattern, response)
        if matches:
            result = matches[0]
        else:
            print("not found!", response)
            result = random.choice(self.label_space)
        return result

    def get_response(self, sample):
        query = self.retriever.get_final_query(sample)
        query = f"Task description: {self.task_description}" + "\n\n" + query
        response = self.get_query_response(query)
        sample.pseudo_label = response
        if self.is_choice:
            sample.demonstration = sample.question + sample.pseudo_label + ")"
        else:
            sample.demonstration = sample.question + sample.pseudo_label
            self.retriever.add_sample(sample)

    def inference(self, sample):
        self.test_sample_num += 1
        self.global_test_sample_num += 1

        self.get_response(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1
            self.global_right_sample_num += 1

    def preprocess(self, sample):
        if self.is_choice:
            prompt = "Q: {question}\nA: ("
        else:
            prompt = "Q: {question}\nA: "
        idx = self.test_sample_num
        question = prompt.format_map({"question": sample["input"]})
        matches = re.findall(self.pattern, sample["target"])
        if matches:
            label = matches[0]
        else:
            print("not a legal label!", sample["target"])
            return None
        embed = self.get_embedding(question).squeeze()
        sample = Sample(idx, question, label, embed, None, None)
        return sample

    def run(self):
        results = {"avg": 0}
        for subset in self.subsets:
            self.pattern = self.subsets[subset]
            self.label_space = self.label_space_map[subset]
            self.task_description = self.task_description_map[subset]
            self.is_choice = (subset in self.is_choices.keys())
            with open(f"bbh/{subset}.json", "r") as f:
                dataset = json.load(f)["examples"]
            self.test_sample_num = 0
            self.right_sample_num = 0
            for idx in tqdm(range(len(dataset)), desc=f"Inference {subset}..."):
                sample = self.preprocess(dataset[idx])
                if sample is None:
                    continue
                else:
                    self.inference(sample)
            acc = self.right_sample_num / self.test_sample_num
            results[subset] = acc
            print(f"{subset}:{acc}")
        results["avg"] = self.global_right_sample_num / self.global_test_sample_num
        return results
