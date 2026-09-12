"""O-LoRA Multi-Adapter (Merged) T2M Evaluation (task-id free inference).

This evaluation differs from `continual_learning/olora_multi_adapter/evaluate_olora_t2m.py`:

- For each stage i, after loading the O-LoRA checkpoint (which stores LoRA weights
  for task adapters `task_0..task_i`), we **merge all learned task adapters** into
  the base model weights via PEFT `merge_and_unload(adapter_names=...)`.

After merging, the model has **no adapters** and therefore inference does not need
task labels.

We still evaluate all tasks j to build the CL matrix R[i,j], but each entry uses
the same merged model for that stage.
"""

from __future__ import annotations

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

_DIR = Path(__file__).parent.resolve()
_CL_DIR = _DIR.parent
_MOTION_AGENT_ROOT = _CL_DIR.parent.resolve()

sys.path.insert(0, str(_MOTION_AGENT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import LoraConfig

from continual_learning.utils.data_manager import TASK_NAMES, get_task_loader, set_split_mode
from models.mllm import MotionLLM
from models.training_utils import process_batch
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from utils.evaluation import evaluation_test
from utils.word_vectorizer import WordVectorizer


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


def _build_display_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for t in TASK_NAMES:
        short = t.replace("task_", "").split("_", 1)
        names[t] = short[1].replace("_", " ").title() if len(short) > 1 else t
    return names


TASK_DISPLAY_NAMES = _build_display_names()
T2M_QUALITY_METRICS = ["fid", "top1", "top2", "top3", "diversity", "mm_dist"]


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

    # Merge pretrained t2m adapter into base weights first.
    if hasattr(args, "pretrained_path") and os.path.exists(args.pretrained_path):
        print(f"  Merging pretrained t2m adapter from {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, "t2m")
        model.llm = model.llm.merge_and_unload()

    # Add task adapters
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
                m = motion[i:i + 1, :m_length[i], :].to(device)
                tok = model.net.encode(m).squeeze(0)
                tok_remapped = torch.from_numpy(model.motion_token_indices[tok.cpu().numpy()]).to(device)
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


def compute_motion_quality_merged(model: MotionLLM, loader, eval_wrapper, temp_dir: str) -> dict:
    """Compute motion quality metrics with a merged model.

    `evaluation_test()` calls `model.generate_batch()`; MotionLLM.generate_batch() calls
    `set_adapter('t2m')` which does not exist after merge. We monkeypatch a minimal
    generate_batch that uses `model.llm.generate()` directly.
    """

    def _generate_batch_no_adapter(captions: List[str], max_length: int = 200):
        if len(captions) == 0:
            return []
        device = next(model.llm.parameters()).device
        model.llm.eval()

        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
        )
        instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        batch_inputs = []
        for caption in captions:
            input_text = "### Input:\n" + caption + "\n\nResponse: <Motion>"
            batch_inputs.append(prompt + instruction + input_text)

        original_padding_side = model.tokenizer.padding_side
        model.tokenizer.padding_side = "left"
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        encoded = model.tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        model.tokenizer.padding_side = original_padding_side

        outputs = model.llm.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=1,
            do_sample=False,
            pad_token_id=model.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        batch_size = len(captions)
        results: List[torch.Tensor] = []
        if len(outputs.scores) > 0:
            scores = torch.stack(outputs.scores)  # [gen_len, batch, vocab]
            for b in range(batch_size):
                sample_scores = scores[:, b, :]
                motion_logits = sample_scores[:, -(model.args.nb_code + 2):]
                motion_tokens = torch.argmax(motion_logits, dim=-1)
                if 1 in motion_tokens:
                    end_idx = motion_tokens.tolist().index(1)
                    motion_tokens = motion_tokens[: max(1, end_idx)]
                motion_tokens = torch.clamp(motion_tokens - 2, min=0)
                results.append(motion_tokens)
        else:
            for _ in range(batch_size):
                results.append(torch.zeros(1, dtype=torch.long, device=device))
        return results

    original_generate_batch = getattr(model, "generate_batch", None)
    model.generate_batch = _generate_batch_no_adapter  # type: ignore
    try:
        fid, div, top1, top2, top3, mm_dist = evaluation_test(
            temp_dir,
            loader,
            model,
            eval_wrapper=eval_wrapper,
            draw=False,
            savenpy=False,
        )
    finally:
        if original_generate_batch is not None:
            model.generate_batch = original_generate_batch  # type: ignore

    return {
        "fid": float(fid),
        "top1": float(top1),
        "top2": float(top2),
        "top3": float(top3),
        "diversity": float(div),
        "mm_dist": float(mm_dist),
    }


def evaluate_task(
    model: MotionLLM,
    task_id: int,
    split: str,
    w_vectorizer,
    args,
    eval_wrapper,
    temp_dir: str | None,
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

    if with_generation and eval_wrapper is not None and temp_dir is not None:
        print("    Generating motions (merged model)...")
        quality = compute_motion_quality_merged(model, loader, eval_wrapper, temp_dir)
        result.update(quality)
        print(
            f"    FID={quality['fid']:.4f}, Top1={quality['top1']:.4f}, "
            f"Div={quality['diversity']:.4f}, MM-Dist={quality['mm_dist']:.4f}"
        )

    return result


def evaluate_stage(
    stage: int,
    checkpoint_path: str,
    exp_dir: Path,
    split: str,
    w_vectorizer,
    args,
    eval_wrapper,
    temp_dir: str | None,
    with_generation: bool,
) -> dict:
    device = torch.device(args.device)
    print(f"\n{'=' * 70}")
    print(f"STAGE {stage} -- {checkpoint_path} (MERGED)")
    print(f"{'=' * 70}")

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
            eval_wrapper=eval_wrapper,
            temp_dir=temp_dir,
            with_generation=with_generation,
        )

    avg = {}
    for k in ["loss", "accuracy"] + T2M_QUALITY_METRICS:
        vals = [per_task_results[t][k] for t in TASK_NAMES if k in per_task_results[t]]
        if vals:
            avg[k] = float(np.mean(vals))

    output = {
        "stage": f"after_task{stage}",
        "checkpoint": str(checkpoint_path),
        "split": split,
        "evaluation_protocol": "T2M O-LoRA (merged adapters; task-id free inference)",
        "task_order": TASK_NAMES,
        "per_task": per_task_results,
        "average": avg,
    }

    out_path = exp_dir / "eval_results"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"after_task{stage}_{split}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_file}")

    del model
    torch.cuda.empty_cache()
    return output


def parse_args():
    p = argparse.ArgumentParser(description="O-LoRA Multi-Adapter (Merged) T2M Evaluation")
    p.add_argument("--exp-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--stage", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--generate-all-stages", action="store_true")

    p.add_argument("--llm-backbone", type=str, default=BACKBONE)
    p.add_argument("--lora-r-t2m", type=int, default=64)
    p.add_argument("--lora-alpha-t2m", type=int, default=64)
    p.add_argument("--lora-r-m2t", type=int, default=32)
    p.add_argument("--lora-alpha-m2t", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--olora-r", type=int, default=64)
    p.add_argument("--olora-alpha", type=int, default=64)
    p.add_argument("--olora-target-modules", type=str, default="qv")

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
    p.add_argument(
        "--pretrained-path",
        type=str,
        default=str(
            Path(__file__).parent.parent.parent.parent.resolve()
            / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"
        ),
    )
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--split-mode", type=str, default="random_80_20", choices=["predefined", "random_80_20"])
    p.add_argument("--split-seed", type=int, default=42)

    args = p.parse_args()
    args.training_task = "t2m"
    args.nb_joints = 22
    args.dataname = "t2m"
    return args


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    with_generation = not args.skip_generation

    set_split_mode(args.split_mode, args.split_seed)
    global TASK_DISPLAY_NAMES
    TASK_DISPLAY_NAMES = _build_display_names()

    print("=" * 70)
    print("O-LoRA Multi-Adapter (Merged) T2M Evaluation")
    print("=" * 70)
    print(f"Experiment: {exp_dir}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print(f"Tasks:      {args.num_tasks}")
    print(f"Generation: {'enabled (final stage only)' if with_generation else 'disabled'}")

    stages = [args.stage] if args.stage is not None else list(range(args.num_tasks))
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
    if with_generation:
        print("Loading EvaluatorModelWrapper for T2M quality metrics...")
        opt_path = str(PRETRAINED / 'checkpoints' / "t2m" / "Comp_v6_KLD005" / "opt.txt")
        wrapper_opt = get_opt(opt_path, args.device)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        temp_dir = tempfile.mkdtemp(prefix="olora_t2m_merged_eval_")
        print(f"  Temp dir: {temp_dir}")

    try:
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
                eval_wrapper=eval_wrapper,
                temp_dir=temp_dir,
                with_generation=with_generation_for_stage,
            )
            all_results[f"after_task{stage}"] = stage_results

        if len(stages) > 1:
            print("\n" + "=" * 70)
            print("TOKEN ACCURACY MATRIX  R[stage, task]  (MERGED)")
            print("=" * 70)
            hdr = f"{'Stage':<12}" + "".join(
                f"{TASK_DISPLAY_NAMES.get(t, t)[:10]:>12}" for t in TASK_NAMES
            )
            print(hdr)
            print("-" * (12 + 12 * args.num_tasks))
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
