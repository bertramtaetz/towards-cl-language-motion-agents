"""O-LoRA Multi-Adapter (Merged) M2T Evaluation (task-id free inference).

Compared to `continual_learning/olora_multi_adapter/evaluate_olora_m2t.py`, this
script merges all learned task adapters into base weights per stage.

After merge, inference does not require task labels because there is no adapter
switching.
"""

from __future__ import annotations

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

_DIR = Path(__file__).parent.resolve()
_CL_DIR = _DIR.parent
_MOTION_AGENT_ROOT = _CL_DIR.parent.resolve()
_PROJECT_ROOT = _MOTION_AGENT_ROOT.parent

sys.path.insert(0, str(_MOTION_AGENT_ROOT))

DATA_ROOT = _PROJECT_ROOT / "datasets" / "HumanML3D" / "HumanML3D"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import LoraConfig

from continual_learning.utils.data_manager import TASK_NAMES, get_task_loader, set_split_mode
from models.mllm import MotionLLM
from models.training_utils import process_batch
from utils.word_vectorizer import WordVectorizer

try:
    from nlgeval import NLGEval

    HAVE_NLGEVAL = True
except ImportError:
    HAVE_NLGEVAL = False
    print("Warning: nlgeval not installed. NLG metrics will be skipped.")

try:
    from bert_score import score as bert_score_fn

    HAVE_BERTSCORE = True
except ImportError:
    HAVE_BERTSCORE = False
    print("Warning: bert_score not installed. BERTScore will be skipped.")


def _resolve_target_modules(mode: str) -> list[str]:
    mode = (mode or "qv").strip().lower()
    if mode in {"qv", "q_proj,v_proj", "q_proj_v_proj"}:
        return ["q_proj", "v_proj"]
    if mode in {"full", "default"}:
        return [
            "o_proj", "q_proj", "up_proj", "v_proj",
            "k_proj", "down_proj", "gate_proj",
        ]
    if "," in mode:
        return [m.strip() for m in mode.split(",") if m.strip()]
    raise ValueError(
        f"Unknown --olora-target-modules={mode}. Use 'qv', 'full', or a comma-separated list."
    )


TASK_DISPLAY_NAMES = {
    "task_1_jumping": "Jumping",
    "task_2_arms_hands": "Arms/Hands",
    "task_3_walking": "Walking",
    "task_4_gestures": "Gestures",
    "task_5_sit_stand": "Sit/Stand",
}

NLG_METRICS = [
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "rouge_l",
    "cider",
    "spice",
    "bertscore",
]


def load_all_references(motion_id: str) -> List[str]:
    text_file = DATA_ROOT / "texts" / f"{motion_id}.txt"
    if not text_file.exists():
        return []
    references = []
    with open(text_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if parts:
                cap = parts[0].strip()
                if cap:
                    references.append(cap)
    return references


def load_olora_checkpoint_merged(checkpoint_path: str, args, device: torch.device, stage: int) -> MotionLLM:
    """Load stage checkpoint, add task adapters, then merge them into base weights."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    task_adapters = set()
    for key in ckpt.keys():
        for part in key.split("."):
            if part.startswith("task_") and part[5:].isdigit():
                task_adapters.add(part)
    task_adapters = sorted(task_adapters)
    print(f"  Found task adapters: {task_adapters}")

    model = MotionLLM(args)
    model = model.to(device)

    if hasattr(args, "pretrained_path") and args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"  Merging pretrained m2t adapter from {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, "m2t")
        model.llm = model.llm.merge_and_unload()

    if task_adapters:
        from peft import get_peft_model as gpm

        task_lora_config = LoraConfig(
            r=args.olora_r,
            lora_alpha=args.olora_alpha,
            target_modules=_resolve_target_modules(args.olora_target_modules),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        first = task_adapters[0]
        model.llm = gpm(model.llm, task_lora_config, adapter_name=first)
        for adapter_name in task_adapters[1:]:
            model.llm.add_adapter(adapter_name, task_lora_config)
        print(f"  Added adapters: {task_adapters}")

    loaded = 0
    for name, param in model.llm.named_parameters():
        if name in ckpt:
            param.data = ckpt[name].to(device, dtype=param.dtype)
            loaded += 1

    if "embeddings" in ckpt:
        model.llm.get_input_embeddings().weight.data[model.nb_text_tokens:] = ckpt["embeddings"].to(device)
        print(f"  Loaded motion embeddings: {ckpt['embeddings'].shape}")
    if "lm_head" in ckpt:
        model.llm.lm_head.weight.data[model.nb_text_tokens:] = ckpt["lm_head"].to(device)
        print(f"  Loaded lm_head: {ckpt['lm_head'].shape}")

    print(f"  Loaded {loaded} LoRA parameters")

    merge_list = [f"task_{i}" for i in range(stage + 1) if f"task_{i}" in task_adapters]
    if not merge_list:
        raise RuntimeError(f"No adapters found to merge at stage={stage}. Found={task_adapters}")
    print(f"  Merging adapters into base weights: {merge_list}")
    model.llm = model.llm.merge_and_unload(adapter_names=merge_list)

    model.eval()
    return model


def compute_token_accuracy_merged(model: MotionLLM, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_losses, all_accs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="    [merged] token acc", leave=False):
            word_emb, pos_oh, caption, sent_len, motion, m_length, tokens, name = batch

            motion_tokens = []
            for i in range(motion.size(0)):
                m = motion[i : i + 1, : m_length[i], :].to(device)
                tok = model.net.encode(m).squeeze(0)
                tok_np = tok.cpu().numpy()
                tok_remapped = torch.from_numpy(model.motion_token_indices[tok_np]).to(device)
                motion_tokens.append(tok_remapped)

            inputs_ids, targets, attention_mask = process_batch(
                tokenizer=model.tokenizer,
                batch_of_captions=list(caption),
                max_tgt_len=200,
                batch_of_motions=motion_tokens,
                training_task="m2t",
            )
            inputs_ids = inputs_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            outputs = model.llm(
                input_ids=inputs_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

            loss = outputs.loss.item()
            chosen = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
            lbls = targets[:, 2:]
            correct = (chosen.reshape(-1) == lbls.reshape(-1)).long()
            valid_mask = (lbls != -100).reshape(-1)
            acc = (correct & valid_mask).sum().item() / (valid_mask.sum().item() + 1.0)

            all_losses.append(loss)
            all_accs.append(acc)

    return float(np.mean(all_losses)), float(np.mean(all_accs))


def generate_caption_merged(model: MotionLLM, motion_np: np.ndarray, device: torch.device) -> str:
    """Generate caption using merged model (no adapters)."""
    model.llm.eval()

    motion_norm = model.normalize(motion_np)
    motion_tensor = torch.from_numpy(motion_norm).float().to(device).unsqueeze(0)
    with torch.no_grad():
        motion_tokens = model.net.encode(motion_tensor).squeeze(0)

    motion_tokens = motion_tokens + model.nb_text_tokens + 2

    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
    )
    instruction = (
        "### Instruction:\nGenerate a caption matching the following input human motion "
        "token sequence.\n\n"
    )
    input_text = (
        "### Input:\n<Motion>" + model.tokenizer.decode(motion_tokens) + "</Motion>\n\nResponse: "
    )
    full_text = prompt + instruction + input_text
    input_ids = model.tokenizer.encode(full_text, return_tensors="pt").to(device)

    with torch.no_grad():
        pred = model.llm.generate(
            input_ids,
            max_length=200,
            num_beams=2,
            pad_token_id=model.tokenizer.pad_token_id,
        )

    pred = pred[0, len(input_ids[0]) :]
    pred_text = model.tokenizer.decode(pred)
    caption = pred_text.split("<eos>")[0].strip()

    caption = re.sub(r"<Motion_\d+>", "", caption)
    caption = re.sub(r"</?Motion>", "", caption)
    return caption.strip()


def compute_bertscore(hypotheses: List[str], references: List[str], device: str) -> float:
    if not HAVE_BERTSCORE or not hypotheses:
        return 0.0
    try:
        _, _, F1 = bert_score_fn(
            hypotheses,
            references,
            lang="en",
            verbose=False,
            device=device if "cuda" in device else None,
        )
        return float(F1.mean().item() * 100)
    except Exception as e:
        print(f"  Warning: BERTScore failed: {e}")
        return 0.0


def evaluate_task(
    model: MotionLLM,
    task_id: int,
    split: str,
    w_vectorizer,
    args,
    nlg_eval,
    with_generation: bool,
) -> dict:
    device = torch.device(args.device)
    task_name = TASK_NAMES[task_id]
    print(f"  Task {task_id} ({task_name}), model=merged")

    loader = get_task_loader(
        dataset_name="t2m",
        split=split,
        batch_size=args.batch_size,
        w_vectorizer=w_vectorizer,
        task_id=task_id,
        num_workers=0,
        unit_length=2 ** args.down_t,
    )
    n_samples = len(loader.dataset)
    print(f"    {n_samples} samples in {split} split")

    loss, accuracy = compute_token_accuracy_merged(model, loader, device)
    print(f"    Loss={loss:.4f}, Accuracy={accuracy:.4f} ({accuracy * 100:.2f}%)")

    result = {
        "task_id": task_id,
        "task_name": task_name,
        "adapter_used": "merged",
        "n_samples": n_samples,
        "loss": loss,
        "accuracy": accuracy,
    }

    if with_generation and nlg_eval is not None:
        print("    Generating captions for NLG metrics (merged model)...")
        predictions: List[str] = []
        references_first: List[str] = []
        references_multi: List[List[str]] = []
        generation_errors = 0

        for batch in tqdm(loader, desc="    [merged] caption gen", leave=False):
            word_emb, pos_oh, caption, sent_len, motion, m_length, tokens, name = batch

            for i in range(motion.size(0)):
                try:
                    m_np = motion[i : i + 1, : m_length[i], :].cpu().numpy()[0]
                    m_denorm = model.mean + m_np * model.std
                    pred = generate_caption_merged(model, m_denorm, device)
                    ref = caption[i]

                    motion_id = str(name[i]) if hasattr(name, "__getitem__") else None
                    refs = load_all_references(motion_id) if motion_id else [ref]
                    if not refs:
                        refs = [ref]
                    if len(refs) > 3:
                        refs = refs[:3]
                    while len(refs) < 3:
                        refs.append(refs[0])

                    predictions.append(pred)
                    references_first.append(refs[0])
                    references_multi.append(refs)
                except Exception:
                    generation_errors += 1
                    predictions.append("")
                    ref = caption[i]
                    references_first.append(ref)
                    references_multi.append([ref, ref, ref])

        result["num_generation_errors"] = generation_errors

        if predictions:
            refs_transposed = [list(refs) for refs in zip(*references_multi)]
            scores = nlg_eval.compute_metrics(refs_transposed, predictions)
            result["bleu1"] = scores["Bleu_1"] * 100
            result["bleu2"] = scores["Bleu_2"] * 100
            result["bleu3"] = scores["Bleu_3"] * 100
            result["bleu4"] = scores["Bleu_4"] * 100
            result["rouge_l"] = scores["ROUGE_L"] * 100
            result["cider"] = scores["CIDEr"] * 100
            if "SPICE" in scores:
                result["spice"] = scores["SPICE"] * 100
            if HAVE_BERTSCORE:
                result["bertscore"] = compute_bertscore(predictions, references_first, args.device)

            print(
                f"    BLEU-4={result.get('bleu4', 0):.2f}, "
                f"ROUGE-L={result.get('rouge_l', 0):.2f}, "
                f"CIDEr={result.get('cider', 0):.2f}, "
                f"BERTScore={result.get('bertscore', 0):.2f}"
            )

    return result


def evaluate_stage(
    stage: int,
    checkpoint_path: str,
    exp_dir: Path,
    split: str,
    w_vectorizer,
    args,
    nlg_eval,
    with_generation: bool,
) -> dict:
    device = torch.device(args.device)
    print("\n" + "=" * 70)
    print(f"STAGE {stage} — {checkpoint_path} (MERGED)")
    print("=" * 70)

    model = load_olora_checkpoint_merged(checkpoint_path, args, device, stage)

    per_task_results = {}
    for task_id in range(args.num_tasks):
        task_name = TASK_NAMES[task_id]
        per_task_results[task_name] = evaluate_task(
            model=model,
            task_id=task_id,
            split=split,
            w_vectorizer=w_vectorizer,
            args=args,
            nlg_eval=nlg_eval,
            with_generation=with_generation,
        )

    avg = {}
    scalar_keys = ["loss", "accuracy"] + NLG_METRICS
    for k in scalar_keys:
        vals = [per_task_results[t][k] for t in TASK_NAMES if k in per_task_results[t]]
        if vals:
            avg[k] = float(np.mean(vals))

    output = {
        "stage": f"after_task{stage}",
        "checkpoint": str(checkpoint_path),
        "split": split,
        "evaluation_protocol": "TM2T (nlgeval with SPICE, multi-reference), merged adapters",
        "task_order": TASK_NAMES,
        "per_task": per_task_results,
        "average": avg,
    }

    out_dir = exp_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"after_task{stage}_{split}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_file}")

    del model
    torch.cuda.empty_cache()
    return output


def parse_args():
    p = argparse.ArgumentParser(description="O-LoRA Multi-Adapter (Merged) M2T Evaluation")
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--generate-all-stages", action="store_true")

    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=32)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--olora-r", type=int, default=32)
    p.add_argument("--olora-alpha", type=int, default=32)
    p.add_argument("--olora-target-modules", type=str, default="qv")

    p.add_argument(
        "--pretrained-path",
        type=str,
        default=str(
            Path(__file__).parent.parent.parent.parent.resolve()
            / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"
        ),
    )
    p.add_argument("--split-mode", type=str, default="random_80_20", choices=["predefined", "random_80_20"])
    p.add_argument("--split-seed", type=int, default=42)

    # VQ-VAE config
    p.add_argument("--nb-code", type=int, default=512)
    p.add_argument("--code-dim", type=int, default=512)
    p.add_argument("--output-emb-width", type=int, default=512)
    p.add_argument("--down-t", type=int, default=2)
    p.add_argument("--stride-t", type=int, default=2)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dilation-growth-rate", type=int, default=3)
    p.add_argument("--vq-act", type=str, default="relu")
    p.add_argument("--vq-norm", type=str, default=None)
    p.add_argument("--quantizer", type=str, default="ema_reset")
    p.add_argument("--mu", type=float, default=0.99)
    p.add_argument("--beta", type=float, default=1.0)

    p.add_argument("--num-tasks", type=int, default=5)

    args = p.parse_args()
    args.training_task = "m2t"
    args.nb_joints = 22
    args.dataname = "t2m"
    return args


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    device = torch.device(args.device)
    with_generation = not args.skip_generation

    set_split_mode(args.split_mode, args.split_seed)

    print("=" * 70)
    print("O-LoRA Multi-Adapter (Merged) M2T Evaluation")
    print("=" * 70)
    print(f"Experiment: {exp_dir}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print(f"Generation: {'enabled (final stage only)' if with_generation else 'disabled'}")
    print(f"Tasks:      {TASK_NAMES}")

    stages = [args.stage] if args.stage is not None else list(range(args.num_tasks))
    ckpt_template = "olora_multi_task_{stage}_best.pth"
    for s in stages:
        ckpt = exp_dir / ckpt_template.format(stage=s)
        if not ckpt.exists():
            print(f"ERROR: checkpoint not found: {ckpt}")
            sys.exit(1)
    print(f"Stages:     {stages}")

    print("\nLoading word vectorizer...")
    glove_path = str(PRETRAINED / 'glove')
    w_vectorizer = WordVectorizer(glove_path, "our_vab")

    nlg_eval = None
    if with_generation and HAVE_NLGEVAL:
        disable_spice = str(os.environ.get("MOTION_AGENT_DISABLE_SPICE", "0")).strip().lower() in {"1", "true", "yes"}
        metrics_to_omit = [
            "METEOR",
            "EmbeddingAverageCosineSimilarity",
            "SkipThoughtCS",
            "VectorExtremaCosineSimilarity",
            "GreedyMatchingScore",
        ]
        if disable_spice:
            metrics_to_omit.append("SPICE")
            print("Loading NLGEval (BLEU, ROUGE-L, CIDEr; SPICE disabled via MOTION_AGENT_DISABLE_SPICE=1)...")
        else:
            print("Loading NLGEval (BLEU, ROUGE-L, CIDEr, SPICE)...")

        nlg_eval = NLGEval(metrics_to_omit=metrics_to_omit)
    elif with_generation and not HAVE_NLGEVAL:
        print("Warning: NLGEval unavailable — NLG metrics will be skipped.")

    all_results = {}
    for stage in stages:
        ckpt_path = exp_dir / ckpt_template.format(stage=stage)
        with_generation_for_stage = with_generation and (
            args.generate_all_stages or (stage == args.num_tasks - 1)
        )
        stage_results = evaluate_stage(
            stage=stage,
            checkpoint_path=str(ckpt_path),
            exp_dir=exp_dir,
            split=args.split,
            w_vectorizer=w_vectorizer,
            args=args,
            nlg_eval=nlg_eval,
            with_generation=with_generation_for_stage,
        )
        all_results[f"after_task{stage}"] = stage_results

    print("\n" + "=" * 70)
    print("TOKEN ACCURACY MATRIX  R[stage, task]  (MERGED)")
    print("=" * 70)
    header = f"{'Stage':<12}" + "".join(
        f"{TASK_DISPLAY_NAMES.get(t, t)[:10]:>12}" for t in TASK_NAMES
    )
    print(header)
    print("-" * (12 + 12 * args.num_tasks))
    for stage in stages:
        row = f"After T{stage}   "
        for task_name in TASK_NAMES:
            acc = all_results[f"after_task{stage}"]["per_task"][task_name]["accuracy"]
            row += f"{acc * 100:>12.2f}"
        print(row)
    print("=" * 70)

    print("\nDone. Run compute_cl_metrics_m2t.py to compute BWT/FWT.")


if __name__ == "__main__":
    main()
