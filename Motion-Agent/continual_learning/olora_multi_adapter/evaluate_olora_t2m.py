"""
O-LoRA Multi-Adapter T2M Evaluation -- Per-Task Adapter Switching

Evaluates all 5 saved checkpoints with per-task adapter switching to build
the full continual learning performance matrix R[i,j].

Protocol:
- For stage i (0..4), load olora_multi_task_i_best.pth
- For each task j (0..4):
    - If j <= i: switch to task_j adapter (dedicated per-task LoRA)
    - If j > i:  switch to t2m base adapter (zero-shot / forward transfer)
  Then compute token accuracy + loss (cheap forward pass)
- At final stage (i=4) only: also generate motions for quality metrics
  (FID, R-Precision, Diversity, MM-Dist)

CRITICAL: Uses model.llm() directly for token accuracy and
model.adapter_override for generation, to avoid model.forward()
overriding the O-LoRA adapter via set_adapter(training_task).

Usage:
    python evaluate_olora_t2m.py \
        --exp-dir experiments/olora_multi/t2m/v1 --split test

    python evaluate_olora_t2m.py \
        --exp-dir experiments/olora_multi/t2m/v1 --split test --skip-generation
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

_OLORA_DIR = Path(__file__).parent.resolve()
_CL_DIR = _OLORA_DIR.parent
_MOTION_AGENT_ROOT = _CL_DIR.parent.resolve()

sys.path.insert(0, str(_MOTION_AGENT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.mllm import MotionLLM
from models.training_utils import process_batch
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.evaluation import evaluation_test
from utils.word_vectorizer import WordVectorizer
from peft import LoraConfig
from continual_learning.utils.data_manager import (
    get_task_loader, TASK_NAMES, set_split_mode,
)

NUM_TASKS = 5


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

def _build_display_names():
    """Build display names from current TASK_NAMES."""
    names = {}
    for t in TASK_NAMES:
        short = t.replace("task_", "").split("_", 1)
        names[t] = short[1].replace("_", " ").title() if len(short) > 1 else t
    return names

TASK_DISPLAY_NAMES = _build_display_names()
T2M_QUALITY_METRICS = ["fid", "top1", "top2", "top3", "diversity", "mm_dist"]


def load_olora_checkpoint(checkpoint_path, args, device):
    """Load MotionLLM with O-LoRA per-task adapters from a checkpoint.

    Uses merge-then-adapt: merges pretrained t2m adapter into base weights
    first, then loads per-task O-LoRA adapters from the checkpoint.
    """
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

    # Merge pretrained adapter into base weights before adding task adapters
    if hasattr(args, 'pretrained_path') and os.path.exists(args.pretrained_path):
        print(f"  Merging pretrained t2m adapter from {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, "t2m")
        # After merge, model.llm has a fresh 't2m' PEFT adapter.
        # Remove it so we can add per-task adapters instead.
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
        print(f"  Loaded embeddings: {ckpt['embeddings'].shape}")
    if "lm_head" in ckpt:
        model.llm.lm_head.weight.data[model.nb_text_tokens:] = ckpt["lm_head"].to(device)
        print(f"  Loaded lm_head: {ckpt['lm_head'].shape}")

    print(f"  Loaded {loaded} LoRA parameters")
    model._task_adapters = task_adapters
    model.eval()
    return model


def compute_token_accuracy(model, loader, device, adapter_name):
    """Compute T2M token accuracy with a specific adapter.

    Uses model.llm() directly to avoid model.forward() overriding the adapter.
    """
    model.eval()
    model.llm.set_adapter(adapter_name)

    all_losses, all_accs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"    [{adapter_name}] token acc", leave=False):
            word_emb, pos_oh, caption, sent_len, motion, m_length, tokens, name = batch

            motion_tokens = []
            for i in range(motion.size(0)):
                m = motion[i:i + 1, :m_length[i], :].to(device)
                tok = model.net.encode(m).squeeze(0)
                tok_remapped = torch.from_numpy(
                    model.motion_token_indices[tok.cpu().numpy()]
                ).to(device)
                motion_tokens.append(tok_remapped)

            inputs_ids, targets, attention_mask = process_batch(
                tokenizer=model.tokenizer,
                batch_of_captions=list(caption),
                max_tgt_len=200,
                batch_of_motions=motion_tokens,
                training_task="t2m",
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


def compute_motion_quality(model, loader, eval_wrapper, temp_dir, adapter_name):
    """Generate motions with a specific adapter and compute quality metrics.

    Sets model.adapter_override so that model.generate_batch() uses the
    correct adapter instead of defaulting to 't2m'.
    """
    model.eval()
    model.adapter_override = adapter_name

    fid, div, top1, top2, top3, mm_dist = evaluation_test(
        temp_dir, loader, model,
        eval_wrapper=eval_wrapper, draw=False, savenpy=False,
    )

    model.adapter_override = None

    return {
        "fid": float(fid), "top1": float(top1), "top2": float(top2),
        "top3": float(top3), "diversity": float(div), "mm_dist": float(mm_dist),
    }


def evaluate_task(model, task_id, stage, split, w_vectorizer, args,
                  eval_wrapper, temp_dir, with_generation):
    """Evaluate one task at a given stage with adapter switching."""
    device = torch.device(args.device)
    task_name = TASK_NAMES[task_id]

    if task_id <= stage:
        adapter_name = f"task_{task_id}"
    else:
        adapter_name = "t2m"

    print(f"  Task {task_id} ({task_name}), adapter={adapter_name}")

    loader = get_task_loader(
        dataset_name="t2m", split=split, batch_size=args.batch_size,
        w_vectorizer=w_vectorizer, task_id=task_id, num_workers=0,
        unit_length=2 ** args.down_t,
    )
    n_samples = len(loader.dataset)
    print(f"    {n_samples} samples in {split} split")

    loss, accuracy = compute_token_accuracy(model, loader, device, adapter_name)
    print(f"    Loss={loss:.4f}, Accuracy={accuracy:.4f} ({accuracy * 100:.2f}%)")

    result = {
        "task_id": task_id, "task_name": task_name, "adapter_used": adapter_name,
        "n_samples": n_samples, "loss": loss, "accuracy": accuracy,
    }

    if with_generation and eval_wrapper is not None:
        print(f"    Generating motions (adapter={adapter_name})...")
        quality = compute_motion_quality(model, loader, eval_wrapper, temp_dir, adapter_name)
        result.update(quality)
        print(f"    FID={quality['fid']:.4f}, Top1={quality['top1']:.4f}, "
              f"Div={quality['diversity']:.4f}, MM-Dist={quality['mm_dist']:.4f}")

    return result


def evaluate_stage(stage, checkpoint_path, exp_dir, split, w_vectorizer, args,
                   eval_wrapper, temp_dir, with_generation):
    device = torch.device(args.device)
    print(f"\n{'=' * 70}")
    print(f"STAGE {stage} -- {checkpoint_path}")
    print(f"{'=' * 70}")

    model = load_olora_checkpoint(checkpoint_path, args, device)

    per_task_results = {}
    for task_id in range(NUM_TASKS):
        task_name = TASK_NAMES[task_id]
        per_task_results[task_name] = evaluate_task(
            model=model, task_id=task_id, stage=stage, split=split,
            w_vectorizer=w_vectorizer, args=args,
            eval_wrapper=eval_wrapper, temp_dir=temp_dir,
            # Generation is controlled at the stage level by the caller.
            # Default pipeline: only final stage generates.
            # Optional: generate metrics for all stages for meaningful BWT/FWT.
            with_generation=with_generation,
        )

    avg = {}
    for k in ["loss", "accuracy"] + T2M_QUALITY_METRICS:
        vals = [result[k] for result in per_task_results.values() if k in result]
        if vals:
            avg[k] = float(np.mean(vals))

    output = {
        "stage": f"after_task{stage}",
        "checkpoint": str(checkpoint_path),
        "split": split,
        "evaluation_protocol": "T2M O-LoRA (per-task adapter switching, Guo et al. metrics)",
        "task_order": list(per_task_results),
        "complete_benchmark": len(per_task_results) == len(TASK_NAMES) == 5,
        "per_task": per_task_results,
        "average": avg,
    }

    print(f"\n  Stage {stage} summary:")
    hdr = f"  {'Task':<25} {'Adapter':<10} {'Loss':>7} {'Acc%':>7}"
    if with_generation:
        hdr += f" {'FID':>8} {'Top1':>7} {'Div':>8} {'MMDist':>8}"
    print(hdr)
    for task_name in per_task_results:
        r = per_task_results[task_name]
        disp = TASK_DISPLAY_NAMES.get(task_name, task_name)
        line = f"  {disp:<25} {r['adapter_used']:<10} {r['loss']:>7.4f} {r['accuracy'] * 100:>7.2f}"
        if with_generation:
            line += (f" {r.get('fid', 0):>8.4f} {r.get('top1', 0):>7.4f}"
                     f" {r.get('diversity', 0):>8.4f} {r.get('mm_dist', 0):>8.4f}")
        print(line)

    out_path = exp_dir / "eval_results"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"after_task{stage}_{split}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(f"\n  Saved: {out_file}")

    del model
    torch.cuda.empty_cache()
    return output


def parse_args():
    p = argparse.ArgumentParser(description="O-LoRA Multi-Adapter T2M Evaluation")
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--generate-all-stages", action="store_true",
                   help="If set, compute motion quality metrics for EVERY stage (very slow). "
                        "Default: only final stage generates motions.")

    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=64)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--olora-r", type=int, default=64,
                   help="LoRA rank for O-LoRA per-task adapters (must match training)")
    p.add_argument("--olora-alpha", type=int, default=64)
    p.add_argument(
        "--olora-target-modules",
        type=str,
        default="qv",
        help="Target modules for O-LoRA per-task adapters: 'qv', 'full', or comma-separated module names.",
    )

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
    p.add_argument("--pretrained-path", type=str,
                   default=str(Path(__file__).parent.parent.parent.parent.resolve()
                               / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"),
                   help="Path to pretrained motionllm.pth for merge-then-adapt")
    p.add_argument("--num-tasks", type=int, default=5,
                   help="Number of tasks")
    p.add_argument("--split-mode", type=str, default="random_80_20",
                   choices=["predefined", "random_80_20"])
    p.add_argument("--split-seed", type=int, default=42)

    args = p.parse_args()
    args.training_task = "t2m"
    args.nb_joints = 22
    args.dataname = "t2m"
    return args


def main():
    global NUM_TASKS, TASK_DISPLAY_NAMES
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    with_generation = not args.skip_generation

    set_split_mode(args.split_mode, args.split_seed)
    NUM_TASKS = args.num_tasks
    TASK_DISPLAY_NAMES = _build_display_names()

    print("=" * 70)
    print("O-LoRA Multi-Adapter T2M Evaluation")
    print("=" * 70)
    print(f"Experiment: {exp_dir}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print(f"Tasks:      {NUM_TASKS}")
    print(f"Generation: {'enabled (final stage only)' if with_generation else 'disabled'}")

    stages = [args.stage] if args.stage is not None else list(range(NUM_TASKS))
    ckpt_template = "olora_multi_task_{stage}_best.pth"
    for s in stages:
        ckpt = exp_dir / ckpt_template.format(stage=s)
        if not ckpt.exists():
            print(f"ERROR: checkpoint not found: {ckpt}")
            sys.exit(1)
    print(f"Stages:     {stages}")

    print("\nLoading word vectorizer...")
    w_vectorizer = WordVectorizer(str(PRETRAINED / 'glove'), "our_vab")

    eval_wrapper = None
    temp_dir = None
    # If we need quality metrics for any stage, we must initialize the evaluator wrapper.
    if with_generation:
        print("Loading EvaluatorModelWrapper for T2M quality metrics...")
        opt_path = str(PRETRAINED / 'checkpoints' / "t2m" / "Comp_v6_KLD005" / "opt.txt")
        wrapper_opt = get_opt(opt_path, args.device)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        temp_dir = tempfile.mkdtemp(prefix="olora_t2m_eval_")
        print(f"  Temp dir: {temp_dir}")

    try:
        all_results = {}
        for stage in stages:
            ckpt_path = exp_dir / ckpt_template.format(stage=stage)

            # Default behaviour: generation only at final stage (stage 4)
            # Optional: enable generation at all stages for meaningful quality-metric BWT/FWT.
            with_generation_for_stage = with_generation and (
                args.generate_all_stages or (stage == NUM_TASKS - 1)
            )
            stage_results = evaluate_stage(
                stage=stage, checkpoint_path=str(ckpt_path),
                exp_dir=exp_dir, split=args.split,
                w_vectorizer=w_vectorizer, args=args,
                eval_wrapper=eval_wrapper, temp_dir=temp_dir,
                with_generation=with_generation_for_stage,
            )
            all_results[f"after_task{stage}"] = stage_results

        if len(stages) > 1:
            print("\n" + "=" * 70)
            print("TOKEN ACCURACY MATRIX  R[stage, task]")
            print("=" * 70)
            hdr = f"{'Stage':<12}" + "".join(
                f"{TASK_DISPLAY_NAMES.get(t, t)[:10]:>12}" for t in TASK_NAMES)
            print(hdr)
            print("-" * (12 + 12 * NUM_TASKS))
            for stage in stages:
                row = f"After T{stage}   "
                for t in TASK_NAMES:
                    acc = all_results[f"after_task{stage}"]["per_task"][t]["accuracy"]
                    row += f"{acc * 100:>12.2f}"
                print(row)
            print("=" * 70)

        print("\nDone.")
        print("Next: compute_cl_metrics_token_acc.py and compute_cl_metrics_t2m.py")

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp dir: {temp_dir}")


if __name__ == "__main__":
    main()
