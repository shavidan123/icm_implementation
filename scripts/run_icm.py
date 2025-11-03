from __future__ import annotations
import os

from src.utils import (
    load_config, seed_everything, ICMDataLoader, make_client,
    accuracy, save_barplot, predict
)
from src.icm import ICM, ICMParams


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "..", "config.yaml")

    cfg = load_config(cfg_path)
    seed_everything(cfg.get("seed", 42))

    api_key = os.getenv("HYPERBOLIC_API_KEY", cfg.get("HYPERBOLIC_API_KEY", ""))
    if not api_key:
        raise RuntimeError("Set HYPERBOLIC_API_KEY in your environment.")
    client = make_client(api_key=api_key, base_url="https://api.hyperbolic.xyz/v1")

    # Limits for pilot runs
    train_limit = cfg["data"].get("train_limit")
    test_limit  = cfg["data"].get("test_limit")

    train = ICMDataLoader(cfg["data"]["train_path"], limit=train_limit)
    test  = ICMDataLoader(cfg["data"]["test_path"],  limit=test_limit)

    print(f"train={len(train)} test={len(test)}  (limits: train={train_limit} test={test_limit})")
    print(f"base={cfg['models']['base_id']}  chat={cfg['models']['chat_id']}")

    test_q = [s["question"] for s in test.samples]
    test_c = [s["claim"]    for s in test.samples]
    gold   = [s["label"]    for s in test.samples]

    print("\nZero-shot (Base)")
    base_preds = predict(client, cfg["models"]["base_id"],  test_q, test_c, demos_ctx=None)
    print(f"Base acc: {accuracy(base_preds, gold):.3f}")

    print("\nZero-shot (Chat)")
    chat_preds = predict(client, cfg["models"]["chat_id"],  test_q, test_c, demos_ctx=None)
    print(f"Chat acc: {accuracy(chat_preds, gold):.3f}")

    print("\nGolden Labels (many-shot on base)")
    gold_ctx: list[str] = []
    max_gold = min(cfg["gold"]["max_ctx_demos"], len(train.samples))
    for s in train.samples[:max_gold]:
        lbl = "True" if s["label"] == 1 else "False"
        gold_ctx += [
            f"Question: {s['question']}",
            f"Claim: {s['claim']}",
            f"I think this claim is {lbl}\n",
        ]
    gold_preds = predict(client, cfg["models"]["base_id"], test_q, test_c, demos_ctx=gold_ctx)
    print(f"Gold acc: {accuracy(gold_preds, gold):.3f}")

    print("\nICM search")
    icm = ICM(
        client,
        cfg["models"]["base_id"],
        ICMParams(
            K_init=cfg["icm"]["K_init"],
            alpha=cfg["icm"]["alpha"],
            T0=cfg["icm"]["T0"],
            Tmin=cfg["icm"]["Tmin"],
            beta=cfg["icm"]["beta"],
            iters=cfg["icm"]["iters"],
            max_ctx_demos=cfg["icm"]["max_ctx_demos"],
            seed=cfg.get("seed", 42),
        ),
    )
    try:
        D_labels = icm.search_labels(train)
    except Exception as e:
        D_labels = getattr(icm, "_last_best", {})
        print(f"search interrupted ({e}); using last best labels: {len(D_labels)}")
    icm_ctx: list[str] = []
    for i, y in list(D_labels.items())[:cfg["icm"]["max_ctx_demos"]]:
        ex = train[i]; lbl = "True" if y == 1 else "False"
        icm_ctx += [
            f"Question: {ex['question']}",
            f"Claim: {ex['claim']}",
            f"I think this claim is {lbl}\n",
        ]
    icm_preds = predict(client, cfg["models"]["base_id"], test_q, test_c, demos_ctx=icm_ctx)

    bars = {
        "Base":  accuracy(base_preds, gold),
        "Chat":  accuracy(chat_preds, gold),
        "ICM":   accuracy(icm_preds, gold),
        "Gold":  accuracy(gold_preds, gold),
    }
    out_plot = cfg.get("results_path", "results/plot_truthfulqa.png")

    save_barplot(
        bars,
        out_plot,
        title="TruthfulQA (Test) — Model Comparison",
        ylabel="Accuracy",
        ylim=(0.0, 1.0),
    )

    print("\n______Accuracy Scores_____")
    for k, v in bars.items():
        print(f"{k:>5}: {v:.3f}")
    print(f"\nFigure saved to: {out_plot}")


if __name__ == "__main__":
    main()
