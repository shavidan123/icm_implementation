# src/icm.py
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict, List

from src.utils import (
    build_truth_prompt,
    is_chat_model,
    consistency_key_of,
    stable_order,
    single_token_toplogprobs,
)
from tqdm import tqdm

@dataclass
class ICMParams:
    K_init: int = 8
    alpha: float = 30.0
    T0: float = 10.0
    Tmin: float = 0.01
    beta: float = 0.99
    iters: int = 200
    max_ctx_demos: int = 256
    seed: int = 42

class ICM:
    """
    Internal Coherence Maximization implementation on TruthfulQA
    """
    def __init__(self, client, base_model_id: str, params: ICMParams):
        self.client = client
        self.model = base_model_id
        self.p = params
        random.seed(self.p.seed)
        self.use_chat_api = is_chat_model(self.model)
        self._last_best: Dict[int, int] = {}

    def _make_context_from_D(self, D: Dict[int, int], dataset, exclude: int | None = None) -> List[str]:
        """
        builds the context from D
        """
        if not D:
            return []
        ctx: List[str] = []
        for i in stable_order([k for k in D.keys() if k != exclude], dataset):
            yi = D[i]
            ex = dataset[i]
            lbl = "True" if yi == 1 else "False"
            ctx += [
                f"Question: {ex['question']}",
                f"Claim: {ex['claim']}",
                f"I think this claim is {lbl}\n",
            ]
        return ctx

    def _label_argmax(self, x_idx: int, D_without_x: Dict[int, int], dataset) -> int:
        """
        choose predicted label
        """
        ex = dataset[x_idx]
        ctx = self._make_context_from_D(D_without_x, dataset, exclude=None)
        prompt = build_truth_prompt(ex["question"], ex["claim"], ctx)
        top_lp = single_token_toplogprobs(self.client, self.model, prompt, self.use_chat_api)
        lp_true  = self._logp_of_label(1, top_lp)
        lp_false = self._logp_of_label(0, top_lp)
        return 1 if lp_true > lp_false else 0

    @staticmethod
    def _logp_of_label(label: int, top_lp: dict) -> float:
        """
        extract log P(token starting with 'True' or 'False') from a top_logprobs dict.
        """
        target = "true" if label == 1 else "false"
        best = -1e30
        for tok, lp in top_lp.items():
            t = tok.strip().lower()
            if t.startswith(target):
                best = max(best, float(lp))
        return best

    def mutual_predictability(self, D: Dict[int, int], dataset) -> float:
        """
        mutual predictability P (part of loss)
        """
        if not D:
            return 0.0

        total = 0.0
        for i, yi in D.items():
            ctx_minus_i = self._make_context_from_D(D, dataset, exclude=i)
            ex = dataset[i]
            prompt = build_truth_prompt(ex["question"], ex["claim"], ctx_minus_i)
            top_lp = single_token_toplogprobs(self.client, self.model, prompt, self.use_chat_api)
            total += self._logp_of_label(yi, top_lp)
        return total / len(D)

    def logical_consistency(self, D: Dict[int, int], dataset) -> int:
        """
        truthfulQA logical consistency kind of from the repo
        """
        inc = 0
        by_q: Dict[int, List[int]] = {}
        for i in D.keys():
            qid = int(dataset[i]["consistency_id"])
            by_q.setdefault(qid, []).append(i)

        for qid, idxs in by_q.items():
            for a in range(len(idxs)):
                i = idxs[a]
                key_i = consistency_key_of(dataset[i])
                yi = D[i]
                for b in range(a + 1, len(idxs)):
                    j = idxs[b]
                    key_j = consistency_key_of(dataset[j])
                    yj = D[j]

                    if key_i != key_j:
                        if yi == 1 and yj == 1:
                            inc += 1
                    else:
                        if yi != yj:
                            inc += 1
        return inc

    def calculate_loss(self, D: Dict[int, int], dataset) -> float:
        """
        U = alpha * P - I for given D
        """
        P = self.mutual_predictability(D, dataset)
        I = self.logical_consistency(D, dataset)
        return self.p.alpha * P - I

    def sample_index(self, all_idxs: List[int], D: Dict[int, int], dataset) -> int:
        """
        Weighted sampling like the repo
        """
        weights = [1.0] * len(all_idxs)
        labeled = set(D.keys())
        for i in labeled:
            qid = int(dataset[i]["consistency_id"])
            for pos, j in enumerate(all_idxs):
                if j not in labeled and int(dataset[j]["consistency_id"]) == qid:
                    weights[pos] = 100.0
        return random.choices(all_idxs, k=1, weights=weights)[0]

    def search_labels(self, train_loader) -> Dict[int, int]:
        """
        search algorithm from the paper
        """
        N = len(train_loader)
        idxs = list(range(N))

        K = min(self.p.K_init, N)
        D: Dict[int, int] = {}
        seed_idxs = random.sample(idxs, k=K)
        seed_vals = [1] * (K // 2) + [0] * (K - K // 2)
        random.shuffle(seed_vals)
        for i, v in zip(seed_idxs, seed_vals):
            D[i] = v

        print(f"ICM init: K={K}, alpha={self.p.alpha}, N={self.p.iters}")
        U_D = self.calculate_loss(D, train_loader)
        best_D = dict(D)
        best_U = U_D
        self._last_best = dict(best_D)
        print(f"ICM initial U(D)={U_D:.3f}")

        for n in tqdm(range(1, self.p.iters + 1), desc="ICM search", leave=True):
            T = max(self.p.Tmin, self.p.T0 / (1.0 + self.p.beta * math.log(n + 1)))
            i = self.sample_index(idxs, D, train_loader)

            D_wo_i = dict(D)
            if i in D_wo_i:
                D_wo_i.pop(i)
            y_hat_i = self._label_argmax(i, D_wo_i, train_loader)

            D_hat = dict(D)
            D_hat[i] = y_hat_i

            U_hat = self.calculate_loss(D_hat, train_loader)
            delta = U_hat - U_D

            accept = (delta > 0) or (random.random() < math.exp(delta / max(self.p.Tmin, T)))
            if accept:
                D = D_hat
                U_D = U_hat
                if U_D > best_U:
                    best_U = U_D
                    best_D = dict(D)
                    self._last_best = dict(best_D)

            if (n % 20 == 0) or (n == self.p.iters):
                inc_now = self.logical_consistency(D, train_loader)
                print(f"ICM iter {n:4d}/{self.p.iters}  T={T:.3f}  U={U_D:.3f}  U*={best_U:.3f}  inc={inc_now}  |D|={len(D)}")

        print("ICM search finished.")
        return best_D
