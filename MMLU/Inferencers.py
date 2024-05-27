import dataclasses
from retriever import DynamicReteiever
from datasets import load_dataset
from tqdm import tqdm
import torch


@dataclasses.dataclass
class Sample:
    idx: int
    question: str
    label: str
    embed: torch.TensorType
    pseudo_label: str or None
    entropy: torch.TensorType or None
    demonstration: str or None


class DAIL:
    """
    Inference code for DAIL. You can inference your data with two steps:
    1). Init:             inferencer = DAIL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, tokenizer, model, sentence_model, subsets, device):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.sentence_model = sentence_model
        self.subsets = subsets
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.label_space = ["A", "B", "C", "D"]
        self.label_space_idx = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i)) for i in self.label_space])
        self.prompt = "Question: {question}\nA.{A}    B.{B}    C.{C}    D.{D}\nAnswer: "
        self.retriever = DynamicReteiever(args)

    def get_embedding(self, sentence):
        embedding = self.sentence_model.encode([sentence], convert_to_tensor=True)
        return embedding

    def get_response(self, sample):
        query = self.retriever.get_final_query(sample)
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids).logits
        outputs = outputs[0][-1].cpu()[self.label_space_idx].squeeze().to(torch.float32)
        response = self.label_space[torch.argmax(outputs).item()]
        entropy = - torch.sum(outputs.softmax(dim=-1) * outputs.log_softmax(dim=-1)).cpu()
        sample.pseudo_label = response
        sample.entropy = entropy
        sample.demonstration = sample.question + sample.pseudo_label
        self.retriever.add_sample(sample)

    def inference(self, sample):
        sample = self.preprocess(sample)
        self.test_sample_num += 1
        self.get_response(sample)
        if sample.pseudo_label == sample.label:
            self.right_sample_num += 1

    def preprocess(self, sample):
        idx = self.test_sample_num
        question = self.prompt.format_map(
            {"question": sample["input"], "A": sample["A"], "B": sample["B"], "C": sample["C"], "D": sample["D"]})
        label = sample["target"]
        embed = self.get_embedding(question).squeeze().cpu()
        sample = Sample(idx, question, label, embed, None, None, None)
        return sample

    def run(self):
        results = {"avg": 0}
        for subset in self.subsets:
            dataset = load_dataset("lukaemon/mmlu", subset)["test"]
            self.test_sample_num = 0
            self.right_sample_num = 0
            for idx in tqdm(range(len(dataset)), desc=f"Inference {subset}..."):
                self.inference(dataset[idx])
            acc = self.right_sample_num / self.test_sample_num
            results[subset] = acc
            results["avg"] += acc
            print(f"{subset}:{acc}")
        results["avg"] /= len(self.subsets)
        return results


class SelfICL:
    """
      You can inference your data with two steps:
      1). Init:             inferencer = SelfICL(**kwargs)
      2). inference:        results = inferencer.run()
      """

    def __init__(self, args, tokenizer, model, subsets, device):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.subsets = subsets
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.label_space = ["A", "B", "C", "D"]
        self.label_space_idx = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i)) for i in self.label_space])
        self.prompt = "Question: {question}\nA.{A}    B.{B}    C.{C}    D.{D}"

    def generate_pseudo_samples(self, query):
        query = 'Following is an example instance, please come up with 4 new, diverse, and creative instances.\n\n' \
                'Example instance:\n{}\n\nNew instance 1:\n'.format(query)
        input_ids = self.tokenizer([query], return_tensors="pt")["input_ids"].to(self.device)
        response = self.model.generate(input_ids, max_new_tokens=1024, do_sample=False)
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)
        response = response.split("\n\n")[2:]
        for i in range(len(response)):
            response[i] = "\n".join(response[i].split("\n")[1:]) + "\nAnswer: "
        return response

    def get_pseudo_labels(self, generated_querys):
        generated_querys.extend([" ", " ", " ", " "])
        generated_querys = generated_querys[:4]
        demonstrations = []
        for i in range(len(generated_querys)):
            inputs = self.tokenizer([generated_querys[i]], return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids).logits
            outputs = outputs[0][-1].cpu()[self.label_space_idx].squeeze()
            label = self.label_space[torch.argmax(outputs).item()]
            demonstrations.append(generated_querys[i] + label)
        return demonstrations

    def get_final_query(self, query):
        generated_querys = self.generate_pseudo_samples(query)
        generated_demonstrations = self.get_pseudo_labels(generated_querys)
        demonstrations = "\n\n".join(generated_demonstrations)
        query = demonstrations + "\n\n" + query + "\nAnswer: "
        return query

    def get_response(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids).logits
        outputs = outputs[0][-1].cpu()[self.label_space_idx].squeeze()
        response = self.label_space[torch.argmax(outputs).item()]
        return response

    def inference(self, sample):
        sample = self.preprocess(sample)
        self.test_sample_num += 1
        query = sample["query"]
        label = sample["label"]
        query = self.get_final_query(query)
        response = self.get_response(query)
        if response == label:
            self.right_sample_num += 1

    def preprocess(self, sample):
        query = self.prompt.format_map(
            {"question": sample["input"], "A": sample["A"], "B": sample["B"], "C": sample["C"], "D": sample["D"]})
        label = sample["target"]
        return {"query": query, "label": label}

    def run(self):
        results = {"avg": 0}
        for subset in self.subsets:
            dataset = load_dataset("lukaemon/mmlu", subset)["test"]
            self.test_sample_num = 0
            self.right_sample_num = 0
            for idx in tqdm(range(len(dataset)), desc=f"Inference {subset}..."):
                self.inference(dataset[idx])
            acc = self.right_sample_num / self.test_sample_num
            results[subset] = acc
            results["avg"] += acc
            print(f"{subset}:{acc}")
        results["avg"] /= len(self.subsets)
        return results


class ZeroShot:
    """
      You can inference your data with two steps:
      1). Init:             inferencer = ZeroShot(**kwargs)
      2). inference:        results = inferencer.run()
      """

    def __init__(self, args, tokenizer, model, subsets, device):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.subsets = subsets
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.label_space = ["A", "B", "C", "D"]
        self.label_space_idx = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i)) for i in self.label_space])
        self.prompt = "Question: {question}\nA.{A}    B.{B}    C.{C}    D.{D}\nAnswer: "

    def get_response(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids).logits
        outputs = outputs[0][-1].cpu()[self.label_space_idx].squeeze()
        response = self.label_space[torch.argmax(outputs).item()]
        return response

    def inference(self, sample):
        sample = self.preprocess(sample)
        self.test_sample_num += 1
        query = sample["query"]
        label = sample["label"]
        response = self.get_response(query)
        if response == label:
            self.right_sample_num += 1

    def preprocess(self, sample):
        query = self.prompt.format_map(
            {"question": sample["input"], "A": sample["A"], "B": sample["B"], "C": sample["C"], "D": sample["D"]})
        label = sample["target"]
        return {"query": query, "label": label}

    def run(self):
        results = {"avg": 0}
        for subset in self.subsets:
            dataset = load_dataset("lukaemon/mmlu", subset)["test"]
            self.test_sample_num = 0
            self.right_sample_num = 0
            for idx in tqdm(range(len(dataset)), desc=f"Inference {subset}..."):
                self.inference(dataset[idx])
            acc = self.right_sample_num / self.test_sample_num
            results[subset] = acc
            results["avg"] += acc
            print(f"{subset}:{acc}")
        results["avg"] /= len(self.subsets)
        return results


class FewShot:
    """
      You can inference your data with two steps:
      1). Init:             inferencer = FewShot(**kwargs)
      2). inference:        results = inferencer.run()
      """

    def __init__(self, args, tokenizer, model, subsets, device):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.subsets = subsets
        self.device = device
        self.test_sample_num = 0
        self.right_sample_num = 0
        self.label_space = ["A", "B", "C", "D"]
        self.label_space_idx = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i)) for i in self.label_space])
        self.prompt = "Question: {question}\nA.{A}    B.{B}    C.{C}    D.{D}\nAnswer: "

    def get_response(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids).logits
        outputs = outputs[0][-1].cpu()[self.label_space_idx].squeeze()
        response = self.label_space[torch.argmax(outputs).item()]
        return response

    def inference(self, sample, query_head):
        sample = self.preprocess(sample)
        self.test_sample_num += 1
        query = sample["query"]
        label = sample["label"]
        response = self.get_response(query_head + query)
        if response == label:
            self.right_sample_num += 1

    def preprocess(self, sample):
        query = self.prompt.format_map(
            {"question": sample["input"], "A": sample["A"], "B": sample["B"], "C": sample["C"], "D": sample["D"]})
        label = sample["target"]
        return {"query": query, "label": label}

    def run(self):
        results = {"avg": 0}
        for subset in self.subsets:
            dataset = load_dataset("lukaemon/mmlu", subset)["test"]
            demonstrations = load_dataset("lukaemon/mmlu", subset)["train"]
            train_samples = []
            for sample in demonstrations:
                train_samples.append(self.prompt.format_map(
                    {"question": sample["input"], "A": sample["A"], "B": sample["B"], "C": sample["C"],
                     "D": sample["D"]}) + sample["target"])
            query_head = "\n\n".join(train_samples) + "\n\n"
            self.test_sample_num = 0
            self.right_sample_num = 0
            for idx in tqdm(range(len(dataset)), desc=f"Inference {subset}..."):
                self.inference(dataset[idx], query_head)
            acc = self.right_sample_num / self.test_sample_num
            results[subset] = acc
            results["avg"] += acc
            print(f"{subset}:{acc}")
        results["avg"] /= len(self.subsets)
        return results
