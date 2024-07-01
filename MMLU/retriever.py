import torch
from rank_bm25 import BM25Okapi
import random


class DynamicReteiever:
    def __init__(self, args):
        self.args = args
        self.demonstrations = []
        self.dnum = 0

    @staticmethod
    def normalize(tensor):
        mean_value = torch.mean(tensor, dim=-1)
        mean_adjusted_tensor = tensor - mean_value
        std_value = torch.std(mean_adjusted_tensor, dim=-1)
        standardized_tensor = mean_adjusted_tensor / std_value
        return standardized_tensor

    def get_final_query(self, sample):
        demonstrations = self.get_demonstrations_from_bank(sample)
        demonstrations = "\n\n".join(demonstrations) + "\n\n" if demonstrations else ""
        query = demonstrations + sample.question
        return query

    def get_demonstrations_from_bank(self, sample):
        self.dnum = min(len(self.demonstrations), 4)
        if self.dnum == 0:
            return []
        if self.args.select_strategy == "random":
            indices = self.get_random(sample)
        elif self.args.select_strategy == "bm25":
            indices = self.get_bm25(sample)
        elif self.args.select_strategy == "topk":
            indices = self.get_topk(sample)
        elif self.args.select_strategy == "dpp":
            indices = self.get_dpp(sample)
        else:
            print("select_strategy is not effective.")
            return
        return [self.demonstrations[i].demonstration for i in indices]

    def get_random(self, sample):
        random_scores = torch.zeros(len(self.demonstrations))
        entropy_scores = self.normalize(torch.stack([sample.entropy for sample in self.demonstrations], dim=0))
        scores = random_scores - self.args.alpha * entropy_scores
        values, indices = torch.topk(scores, self.dnum, largest=True)
        indices = indices.tolist()
        indices.reverse()
        return indices

    def get_bm25(self, sample):
        tokenized_examples = [example.question.split() for example in self.demonstrations]
        tokenized_query = sample.question.split()
        bm25 = BM25Okapi(tokenized_examples)
        bm25_scores = torch.tensor(bm25.get_scores(tokenized_query))
        bm25_scores = self.normalize(bm25_scores)
        entropy_scores = self.normalize(torch.stack([sample.entropy for sample in self.demonstrations], dim=0))
        scores = bm25_scores - self.args.alpha * entropy_scores
        values, indices = torch.topk(scores, self.dnum, largest=True)
        indices = indices.tolist()
        indices.reverse()
        return indices

    def get_topk(self, sample):
        demonstration_embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
        topk_scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        topk_scores = self.normalize(topk_scores)
        entropy_scores = self.normalize(torch.stack([sample.entropy for sample in self.demonstrations], dim=0))
        scores = topk_scores - self.args.alpha * entropy_scores
        values, indices = torch.topk(scores, self.dnum, largest=True)
        indices = indices.tolist()
        indices.reverse()
        return indices

    def get_dpp(self, sample):
        demonstration_embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
        topk_scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        normalized_topk_scores = self.normalize(topk_scores)
        entropy_scores = self.normalize(torch.stack([sample.entropy for sample in self.demonstrations], dim=0))
        scores = normalized_topk_scores - self.args.alpha * entropy_scores
        values, indices = torch.topk(scores, min(len(self.demonstrations), self.args.dpp_candidates), largest=True)
        candidates = demonstration_embeds[indices]
        near_reps, rel_scores, kernel_matrix = self.get_kernel(sample.embed, candidates)
        samples_ids = torch.tensor(self.fast_map_dpp(kernel_matrix))
        samples_scores = rel_scores[samples_ids]
        _, ordered_indices = torch.sort(samples_scores, descending=False)
        indices = indices[samples_ids[ordered_indices]]
        return indices

    def get_kernel(self, embed, candidates):
        near_reps = candidates
        embed = embed / torch.linalg.norm(embed)
        near_reps = near_reps / torch.linalg.norm(near_reps, keepdims=True, dim=1)
        rel_scores = torch.matmul(embed, near_reps.T)
        rel_scores = (rel_scores + 1) / 2
        rel_scores -= rel_scores.max()
        rel_scores = torch.exp(rel_scores / (2 * self.args.scale_factor))
        sim_matrix = torch.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1) / 2
        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix

    def fast_map_dpp(self, kernel_matrix):
        item_size = kernel_matrix.size()[0]
        cis = torch.zeros([self.dnum, item_size])
        di2s = torch.diag(kernel_matrix)
        selected_items = list()
        selected_item = torch.argmax(di2s)
        selected_items.append(int(selected_item))
        while len(selected_items) < self.dnum:
            k = len(selected_items) - 1
            ci_optimal = cis[:k, selected_item]
            di_optimal = torch.sqrt(di2s[selected_item])
            elements = kernel_matrix[selected_item, :]
            eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
            cis[k, :] = eis
            di2s -= torch.square(eis)
            selected_item = torch.argmax(di2s)
            selected_items.append(int(selected_item))
        return selected_items

    def resize(self):
        if self.args.exit_strategy == "random":
            indices = torch.tensor(random.choices(range(self.args.K), k=self.args.K // 2))
        elif self.args.exit_strategy == "fifo":
            indices = torch.tensor(range(self.args.K // 2,self.args.K))
        elif self.args.exit_strategy == "diverse":
            embeds = torch.stack([sample.embed for sample in self.demonstrations], dim=0)
            sim_scores = []
            for i in range(embeds.size(0)):
                sim_scores.append(torch.mean(torch.cosine_similarity(embeds[i], embeds, dim=-1)))
            sim_scores = torch.stack(sim_scores, dim=0)
            sim_scores = self.normalize(sim_scores)
            entropy_scores = self.normalize(torch.stack([sample.entropy for sample in self.demonstrations], dim=0))
            scores = sim_scores - self.args.alpha * entropy_scores
            values, indices = torch.topk(scores, k=self.args.K // 2, largest=False)
        else:
            print("exit_strategy is not effective.")
            return
        self.demonstrations = [self.demonstrations[i] for i in indices]

    def add_sample(self, sample):
        sample.demonstration = sample.question + sample.pseudo_label
        self.demonstrations.append(sample)
        if len(self.demonstrations) == self.args.M:
            self.resize()
