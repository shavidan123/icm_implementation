from __future__ import annotations
import json, random, math
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple

import yaml
import matplotlib.pyplot as plt
import openai
import time
import os
from tqdm import tqdm


class ICMDataLoader:
    """
    data loader (really just meant for the truthfulqa dataset)
    format is smth like: {"question": str, "choice": str, "label": 0/1, "consistency_id": int}
    """
    def __init__(self, filepath: str | Path, limit: Optional[int] = None):
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        if limit is not None:
            data = data[:limit]
        self.samples: List[Dict] = [
            {
                "question": str(r["question"]).strip(),
                "claim":    str(r["choice"]).strip(),
                "label":    int(r["label"]),
                "consistency_id": int(r["consistency_id"]),
            } for r in data
        ]

    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, i: int) -> Dict: return self.samples[i]
    def __iter__(self) -> Iterable[Dict]: return iter(self.samples)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def seed_everything(seed: int) -> None:
    random.seed(seed)

def accuracy(preds: List[int], gold: List[int]) -> float:
    return sum(int(p == g) for p, g in zip(preds, gold)) / max(1, len(gold))

def save_barplot(
    bars: Dict[str, float],
    out_path: str,
    *,
    title: str = "TruthfulQA Accuracy",
    ylabel: str = "Accuracy",
    ylim: tuple[float, float] = (0.0, 1.0),
    colors: Optional[Iterable[str]] = None,
    annotate: bool = True,
    dpi: int = 200,
) -> str:
    """
    Save a clean barplot from a mapping of label -> score in [0,1].

    Args:
        bars: dict like {"Base": 0.51, "Chat": 0.56, ...}
        out_path: file path to save (e.g., "figs/truthfulqa_icm.png")
        title, ylabel: text labels
        ylim: y-axis range
        colors: iterable of color specs (hex or named). If None, a nice default is used.
        annotate: write value labels on top of bars
        dpi: output resolution

    Returns:
        The path saved to (same as out_path).
    """
    default_palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    if colors is None:
        colors = default_palette

    labels = list(bars.keys())
    values = [bars[k] for k in labels]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure(figsize=(6.5, 4.0))
    ax = plt.gca()

    bar_containers = ax.bar(
        range(len(labels)),
        values,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        edgecolor="none",
        width=0.6,
    )

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if annotate:
        for rect, v in zip(bar_containers, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.02,
                f"{v*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def make_client(api_key: str, base_url: str):
    return openai.OpenAI(api_key=api_key, base_url=base_url)

#kinda copied this from the original repository
def build_truth_prompt(question: str, claim: str, ctx: List[str]) -> str:
    lines: List[str] = []
    if ctx:
        lines += ctx
    lines += [
        f"Question: {question}",
        f"Claim: {claim}",
        "I think this claim is "
    ]
    return "\n".join(lines)

def consistency_key_of(sample: dict) -> str:
    return 'A' if int(sample["label"]) == 1 else 'B'

def stable_order(indices: List[int], dataset) -> List[int]:
    return sorted(indices, key=lambda i: (int(dataset[i]["consistency_id"]), i))

def is_chat_model(model_id: str) -> bool:
    return "instruct" in model_id.lower()


def toplogprobs_dict(resp, use_chat: bool) -> dict:
    if use_chat:
        lp = resp.choices[0].logprobs
        if lp is None:
            content = (resp.choices[0].message.content or "").strip().lower()
            is_true = content.startswith("true")
            return {" True": 0.0 if is_true else -1e9, " False": 0.0 if not is_true else -1e9}

        tlp = lp.content[0].top_logprobs
        if isinstance(tlp, dict):
            return tlp
        out = {}
        for t in tlp:
            out[t["token"]] = float(t["logprob"])
        return out
    else:
        return resp.choices[0].logprobs.top_logprobs[0]


def yesno_logdiff(d0: dict) -> float:
    eps = 1e-5
    p_true = eps
    p_false = eps
    for tok, lp in d0.items():
        t = tok.lower()
        if "true" in t and "false" in t:
            continue
        if "true" in t:
            p_true += math.exp(lp)
        elif "false" in t:
            p_false += math.exp(lp)
    if p_true == eps and p_false == eps:
        return 0.0
    return math.log(p_true) - math.log(p_false)

def single_token_toplogprobs(
    client,
    model: str,
    prompt: str,
    use_chat_api: bool,
) -> dict:
    """
    fetches the top-logprobs dict for the first generated token (added retries bc api was flaky).
    """
    tries = 0
    while True:
        tries += 1
        try:
            if use_chat_api:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                )
            else:
                resp = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=20,
                )
            return toplogprobs_dict(resp, use_chat_api)
        except Exception as e:
            if tries >= 3:
                print(f"api giving up after {tries} tries: {e}")
                raise
            sleep = 0.7 * tries
            print(f"api retry {tries} after error: {e} (sleep {sleep:.2f}s)")
            time.sleep(sleep)

def true_false_with_score(
    client,
    model: str,
    prompt: str,
    *,
    use_chat_api: bool,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[int, float]:
    tries = 0
    while True:
        tries += 1
        try:
            d0 = single_token_toplogprobs(client, model, prompt, use_chat_api)
            break
        except Exception as e:
            if tries >= 3:
                print(f"api giving up after {tries} tries: {e}")
                raise
            sleep = 0.7 * tries
            print(f"api retry {tries} after error: {e} (sleep {sleep:.2f}s)")
            time.sleep(sleep)

    score = yesno_logdiff(d0)
    label = 1 if score > 0 else 0
    return label, score

def predict(client, model_id: str, q: list[str], c: list[str], *, demos_ctx: list[str] | None) -> list[int]:
    preds: list[int] = []
    use_chat = is_chat_model(model_id)
    n = len(q)
    
    ctx_ex = (len(demos_ctx or []) // 3)
    print(f"[predict] model={model_id} ctx_examples={ctx_ex} n={n}")
    for i in tqdm(range(n), desc="[predict]", leave=True):
        qq, cc = q[i], c[i]
        prompt = build_truth_prompt(qq, cc, demos_ctx or [])
        y, _ = true_false_with_score(client, model_id, prompt, use_chat_api=use_chat, temperature=0.0, top_p=1.0)
        preds.append(int(y))
    return preds
