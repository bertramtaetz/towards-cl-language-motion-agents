"""
O-LoRA Multi-Adapter M2T Evaluation — Per-Task Adapter Switching

Evaluates all 5 saved checkpoints with per-task adapter switching to build
the full continual learning performance matrix R[i,j].

Protocol:
- For stage i (0..4), load olora_multi_task_i_best.pth
- For each task j (0..4):
    - If j <= i: switch to task_j adapter (the dedicated per-task LoRA)
    - If j > i:  switch to m2t base adapter (zero-shot / forward transfer)
  Then compute token accuracy + loss (cheap forward pass)
- At final stage (i=4) only: also generate captions for NLG metrics
  (BLEU, ROUGE-L, CIDEr, SPICE, BERTScore)

Outputs:
  <exp-dir>/eval_results/after_task{N}_test.json  for N in 0..4
    - Contains per_task dict keyed by task name
    - Format compatible with compute_cl_metrics_m2t.py

Usage:
    # Evaluate all 5 stages (token accuracy + final-stage NLG metrics)
    python evaluate_olora_m2t.py \\
        --exp-dir experiments/olora_multi/m2t/v1 \\
        --split test

    # Token accuracy only (skip generation, much faster)
    python evaluate_olora_m2t.py \\
        --exp-dir experiments/olora_multi/m2t/v1 \\
        --split test --skip-generation

    # Single stage
    python evaluate_olora_m2t.py \\
        --exp-dir experiments/olora_multi/m2t/v1 \\
        --split test --stage 4
"""

from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_OLORA_DIR = Path(__file__).parent.resolve()
_CL_DIR = _OLORA_DIR.parent
_MOTION_AGENT_ROOT = _CL_DIR.parent.resolve()
_PROJECT_ROOT = _MOTION_AGENT_ROOT.parent

sys.path.insert(0, str(_MOTION_AGENT_ROOT))

# Data root for multi-reference captions (TM2T protocol)
DATA_ROOT = _PROJECT_ROOT / "datasets" / "HumanML3D" / "HumanML3D"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ---------------------------------------------------------------------------
# Imports after path setup
# ---------------------------------------------------------------------------

from models.mllm import MotionLLM
from models.training_utils import process_batch
from peft import LoraConfig
from utils.word_vectorizer import WordVectorizer
from continual_learning.utils.data_manager import (
    get_task_loader,
    TASK_NAMES,
    set_split_mode,
)

# nlgeval — standardized TM2T protocol
try:
    from nlgeval import NLGEval
    HAVE_NLGEVAL = True
except ImportError:
    HAVE_NLGEVAL = False
    print("Warning: nlgeval not installed. NLG metrics will be skipped.")

# BERTScore
try:
    from bert_score import score as bert_score_fn
    HAVE_BERTSCORE = True
except ImportError:
    HAVE_BERTSCORE = False
    print("Warning: bert_score not installed. BERTScore will be skipped.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_TASKS = len(TASK_NAMES)


def _resolve_target_modules(mode: str) -> List[str]:
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
    'task_1_jumping':    'Jumping',
    'task_2_arms_hands': 'Arms/Hands',
    'task_3_walking':    'Walking',
    'task_4_gestures':   'Gestures',
    'task_5_sit_stand':  'Sit/Stand',
}
NLG_METRICS = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge_l', 'cider', 'spice', 'bertscore']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_olora_checkpoint(checkpoint_path: str, args, device: torch.device) -> MotionLLM:
    """
    Load MotionLLM with O-LoRA per-task adapters from a checkpoint.

    Strategy:
    1. Inspect the checkpoint to find which task adapters exist (task_0..task_N)
    2. Build a fresh MotionLLM (creates t2m + m2t adapters)
    3. Add matching LoRA adapter configs for each task adapter found
    4. Load all weights from checkpoint
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Detect which task adapters are present
    task_adapters = set()
    for key in ckpt.keys():
        for part in key.split('.'):
            if part.startswith('task_') and part[5:].isdigit():
                task_adapters.add(part)
    task_adapters = sorted(task_adapters)
    print(f"  Found task adapters: {task_adapters}")

    # Build fresh model
    model = MotionLLM(args)
    model = model.to(device)

    # Merge pretrained m2t direction adapter into base weights before adding task adapters
    if hasattr(args, "pretrained_path") and args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"  Merging pretrained m2t adapter from {args.pretrained_path}")
        model.merge_pretrained_adapter(args.pretrained_path, "m2t")
        # After merge, model.llm has a fresh 'm2t' PEFT adapter.
        # Remove it so we can add per-task adapters instead.
        model.llm = model.llm.merge_and_unload()

    # Add per-task LoRA adapters (qv vs full selectable via --olora-target-modules)
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

    # Load weights
    loaded = 0
    for name, param in model.llm.named_parameters():
        if name in ckpt:
            param.data = ckpt[name].to(device, dtype=param.dtype)
            loaded += 1

    # Load motion token embeddings and lm_head
    if 'embeddings' in ckpt:
        model.llm.get_input_embeddings().weight.data[model.nb_text_tokens:] = \
            ckpt['embeddings'].to(device)
        print(f"  Loaded motion embeddings: {ckpt['embeddings'].shape}")
    if 'lm_head' in ckpt:
        model.llm.lm_head.weight.data[model.nb_text_tokens:] = \
            ckpt['lm_head'].to(device)
        print(f"  Loaded lm_head: {ckpt['lm_head'].shape}")

    print(f"  Loaded {loaded} LoRA parameters")

    # Store available adapters on model for easy reference
    model._task_adapters = task_adapters

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Token accuracy computation
# ---------------------------------------------------------------------------

def compute_token_accuracy(
    model: MotionLLM,
    loader,
    device: torch.device,
    adapter_name: str,
) -> Tuple[float, float]:
    """
    Compute token-level loss and accuracy for one task with a specific adapter.

    Uses model.llm() directly to bypass MotionLLM.forward()'s adapter switching.
    The adapter is set explicitly before the forward pass.

    Returns:
        (mean_loss, mean_accuracy)
    """
    model.eval()
    model.llm.set_adapter(adapter_name)

    all_losses = []
    all_accs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"    [{adapter_name}] token acc", leave=False):
            word_emb, pos_oh, caption, sent_len, motion, m_length, tokens, name = batch

            # Encode motion to VQ-VAE tokens
            motion_tokens = []
            for i in range(motion.size(0)):
                m = motion[i:i+1, :m_length[i], :].to(device)
                with torch.no_grad():
                    tok = model.net.encode(m).squeeze(0)
                tok_np = tok.cpu().numpy()
                tok_remapped = torch.from_numpy(
                    model.motion_token_indices[tok_np]
                ).to(device)
                motion_tokens.append(tok_remapped)

            # Build M2T batch (motion as input, caption as target)
            inputs_ids, targets, attention_mask = process_batch(
                tokenizer=model.tokenizer,
                batch_of_captions=list(caption),
                max_tgt_len=200,
                batch_of_motions=motion_tokens,
                training_task='m2t',
            )
            inputs_ids = inputs_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Forward pass through LLM directly (adapter already set above)
            outputs = model.llm(
                input_ids=inputs_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

            loss = outputs.loss.item()

            # Token accuracy (same formula as training)
            chosen = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
            lbls = targets[:, 2:]
            correct = (chosen.reshape(-1) == lbls.reshape(-1)).long()
            valid_mask = (lbls != -100).reshape(-1)
            acc = (correct & valid_mask).sum().item() / (valid_mask.sum().item() + 1.0)

            all_losses.append(loss)
            all_accs.append(acc)

    return float(np.mean(all_losses)), float(np.mean(all_accs))


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def load_all_references(motion_id: str) -> List[str]:
    """Load all reference captions for a motion ID (TM2T multi-reference protocol)."""
    text_file = DATA_ROOT / "texts" / f"{motion_id}.txt"
    if not text_file.exists():
        return []
    references = []
    with open(text_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('#')
            if parts:
                cap = parts[0].strip()
                if cap:
                    references.append(cap)
    return references


def generate_caption_with_adapter(
    model: MotionLLM,
    motion_np: np.ndarray,
    adapter_name: str,
    device: torch.device,
) -> str:
    """
    Generate a caption from a raw (unnormalized) motion array using a specific adapter.

    Bypasses MotionLLM.caption() which hard-codes set_adapter('m2t').
    Mirrors the same M2T prompt format.
    """
    model.llm.set_adapter(adapter_name)
    model.llm.eval()

    motion_norm = model.normalize(motion_np)
    motion_tensor = torch.from_numpy(motion_norm).float().to(device).unsqueeze(0)

    with torch.no_grad():
        motion_tokens = model.net.encode(motion_tensor).squeeze(0)

    # Reindex to vocab space
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
        "### Input:\n<Motion>"
        + model.tokenizer.decode(motion_tokens)
        + "</Motion>\n\nResponse: "
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

    pred = pred[0, len(input_ids[0]):]
    pred_text = model.tokenizer.decode(pred)
    caption = pred_text.split('<eos>')[0].strip()

    # Filter leaked motion tokens
    caption = re.sub(r'<Motion_\d+>', '', caption)
    caption = re.sub(r'</?Motion>', '', caption)
    return caption.strip()


def compute_bertscore(hypotheses: List[str], references: List[str], device: str) -> float:
    """Compute mean BERTScore F1 (×100)."""
    if not HAVE_BERTSCORE or not hypotheses:
        return 0.0
    try:
        _, _, F1 = bert_score_fn(
            hypotheses, references, lang='en', verbose=False, model_type=str(PRETRAINED / 'roberta-large'), num_layers=17,
            device=device if 'cuda' in device else None,
        )
        return float(F1.mean().item() * 100)
    except Exception as e:
        print(f"  Warning: BERTScore failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    model: MotionLLM,
    task_id: int,          # 0-indexed
    stage: int,            # which stage checkpoint we loaded
    split: str,
    w_vectorizer,
    args,
    nlg_eval,
    with_generation: bool,
) -> dict:
    """
    Evaluate one task at a given stage.

    Adapter selection:
    - task_id <= stage  →  use 'task_{task_id}' (dedicated adapter for that task)
    - task_id >  stage  →  use 'm2t' (zero-shot / forward transfer; no dedicated adapter yet)
    """
    device = torch.device(args.device)
    task_name = TASK_NAMES[task_id]

    if task_id <= stage:
        adapter_name = f"task_{task_id}"
    else:
        adapter_name = 'm2t'

    print(f"  Task {task_id} ({task_name}), adapter={adapter_name}")

    loader = get_task_loader(
        dataset_name='t2m',
        split=split,
        batch_size=args.batch_size,
        w_vectorizer=w_vectorizer,
        task_id=task_id,
        num_workers=0,
        unit_length=2 ** args.down_t,
    )

    n_samples = len(loader.dataset)
    print(f"    {n_samples} samples in {split} split")

    # --- Token accuracy ---
    loss, accuracy = compute_token_accuracy(model, loader, device, adapter_name)
    print(f"    Loss={loss:.4f}, Accuracy={accuracy:.4f} ({accuracy*100:.2f}%)")

    result = {
        'task_id': task_id,
        'task_name': task_name,
        'adapter_used': adapter_name,
        'n_samples': n_samples,
        'loss': loss,
        'accuracy': accuracy,
    }

    # --- Caption generation + NLG metrics (optional) ---
    if with_generation and nlg_eval is not None:
        print(f"    Generating captions for NLG metrics...")
        predictions = []
        references_first = []
        references_multi = []
        generation_errors = 0

        model.llm.set_adapter(adapter_name)

        for batch in tqdm(loader, desc=f"    [{adapter_name}] caption gen", leave=False):
            word_emb, pos_oh, caption, sent_len, motion, m_length, tokens, name = batch

            for i in range(motion.size(0)):
                try:
                    m_np = motion[i:i+1, :m_length[i], :].cpu().numpy()[0]
                    m_denorm = model.mean + m_np * model.std  # denormalize
                    pred = generate_caption_with_adapter(model, m_denorm, adapter_name, device)
                    ref = caption[i]

                    # Multi-reference from dataset files (TM2T protocol)
                    motion_id = str(name[i]) if hasattr(name, '__getitem__') else None
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

                except Exception as e:
                    generation_errors += 1
                    predictions.append("")
                    ref = caption[i]
                    references_first.append(ref)
                    references_multi.append([ref, ref, ref])

        result['num_generation_errors'] = generation_errors

        # NLG metrics via nlgeval
        if predictions:
            refs_transposed = [list(refs) for refs in zip(*references_multi)]
            scores = nlg_eval.compute_metrics(refs_transposed, predictions)
            result['bleu1'] = scores['Bleu_1'] * 100
            result['bleu2'] = scores['Bleu_2'] * 100
            result['bleu3'] = scores['Bleu_3'] * 100
            result['bleu4'] = scores['Bleu_4'] * 100
            result['rouge_l'] = scores['ROUGE_L'] * 100
            result['cider'] = scores['CIDEr'] * 100
            if 'SPICE' in scores:
                result['spice'] = scores['SPICE'] * 100

            # BERTScore
            if HAVE_BERTSCORE:
                result['bertscore'] = compute_bertscore(
                    predictions, references_first, args.device
                )

            print(f"    BLEU-4={result.get('bleu4', 0):.2f}, "
                  f"ROUGE-L={result.get('rouge_l', 0):.2f}, "
                  f"CIDEr={result.get('cider', 0):.2f}, "
                  f"BERTScore={result.get('bertscore', 0):.2f}")

    return result


# ---------------------------------------------------------------------------
# Per-stage evaluation
# ---------------------------------------------------------------------------

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
    """
    Load checkpoint for a stage and evaluate all 5 tasks.
    Saves results to <exp_dir>/eval_results/after_task{stage}_{split}.json
    """
    device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"STAGE {stage} — {checkpoint_path}")
    print(f"{'='*70}")

    model = load_olora_checkpoint(checkpoint_path, args, device)

    per_task_results = {}
    for task_id in range(NUM_TASKS):
        task_name = TASK_NAMES[task_id]
        task_result = evaluate_task(
            model=model,
            task_id=task_id,
            stage=stage,
            split=split,
            w_vectorizer=w_vectorizer,
            args=args,
            nlg_eval=nlg_eval,
            # Generation is controlled at the stage level by the caller.
            # Default pipeline: only final stage generates.
            # Optional: generate metrics for all stages for meaningful NLG BWT/FWT.
            with_generation=with_generation,
        )
        per_task_results[task_name] = task_result

    # Compute averages
    avg = {}
    scalar_keys = ['loss', 'accuracy'] + NLG_METRICS
    for k in scalar_keys:
        vals = [per_task_results[t][k] for t in TASK_NAMES if k in per_task_results[t]]
        if vals:
            avg[k] = float(np.mean(vals))

    # Build output document
    output = {
        'stage': f'after_task{stage}',
        'checkpoint': str(checkpoint_path),
        'split': split,
        'evaluation_protocol': 'TM2T (nlgeval with SPICE, multi-reference), per-task adapter',
        'task_order': TASK_NAMES,
        'per_task': per_task_results,
        'average': avg,
    }

    # Print summary
    print(f"\n  Stage {stage} summary:")
    print(f"  {'Task':<25} {'Adapter':<12} {'Loss':>7} {'Acc%':>7}", end='')
    if with_generation:
        print(f" {'BLEU4':>7} {'ROUGE-L':>8} {'BERTScr':>8}", end='')
    print()
    for task_name in TASK_NAMES:
        r = per_task_results[task_name]
        disp = TASK_DISPLAY_NAMES.get(task_name, task_name)
        print(f"  {disp:<25} {r['adapter_used']:<12} {r['loss']:>7.4f} {r['accuracy']*100:>7.2f}", end='')
        if with_generation:
            print(f" {r.get('bleu4', 0):>7.2f} {r.get('rouge_l', 0):>8.2f} {r.get('bertscore', 0):>8.2f}", end='')
        print()

    # Save
    out_dir = exp_dir / 'eval_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'after_task{stage}_{split}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_file}")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='O-LoRA Multi-Adapter M2T Per-Task Evaluation'
    )

    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory containing olora_multi_task_N_best.pth')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--stage', type=int, default=None,
                        help='Evaluate single stage (0-4). Default: all stages.')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip caption generation (token accuracy only, much faster)')
    parser.add_argument('--generate-all-stages', action='store_true',
                        help='If set, compute NLG metrics for EVERY stage (very slow). '
                             'Default: only final stage generates captions.')

    # Model config — must match training
    parser.add_argument('--llm-backbone', type=str, default=BACKBONE)
    parser.add_argument('--lora-r-t2m', type=int, default=32)
    parser.add_argument('--lora-alpha-t2m', type=int, default=64)
    parser.add_argument('--lora-r-m2t', type=int, default=32)
    parser.add_argument('--lora-alpha-m2t', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)

    parser.add_argument('--olora-r', type=int, default=32,
                        help='LoRA rank for O-LoRA per-task adapters (must match training)')
    parser.add_argument('--olora-alpha', type=int, default=32,
                        help='LoRA alpha for O-LoRA per-task adapters (must match training)')

    parser.add_argument(
        "--olora-target-modules",
        type=str,
        default="qv",
        help="Target modules for O-LoRA per-task adapters: 'qv', 'full', or comma-separated module names.",
    )

    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=str(
            Path(__file__).parent.parent.parent.parent.resolve()
            / "experiments" / "pretrained" / "motionllm_fixed_v1" / "motionllm.pth"
        ),
        help="Path to pretrained motionllm.pth for merge-then-adapt",
    )

    parser.add_argument(
        "--split-mode",
        type=str,
        default="random_80_20",
        choices=["predefined", "random_80_20"],
    )
    parser.add_argument("--split-seed", type=int, default=42)

    # VQ-VAE config
    parser.add_argument('--nb-code', type=int, default=512)
    parser.add_argument('--code-dim', type=int, default=512)
    parser.add_argument('--output-emb-width', type=int, default=512)
    parser.add_argument('--down-t', type=int, default=2)
    parser.add_argument('--stride-t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation-growth-rate', type=int, default=3)
    parser.add_argument('--vq-act', type=str, default='relu')
    parser.add_argument('--vq-norm', type=str, default=None)
    parser.add_argument('--quantizer', type=str, default='ema_reset')
    parser.add_argument('--mu', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()
    args.training_task = 'm2t'
    args.nb_joints = 22
    args.dataname = 't2m'
    return args


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    device = torch.device(args.device)
    with_generation = not args.skip_generation

    set_split_mode(args.split_mode, args.split_seed)

    print("=" * 70)
    print("O-LoRA Multi-Adapter M2T Per-Task Evaluation")
    print("=" * 70)
    print(f"Experiment: {exp_dir}")
    print(f"Split:      {args.split}")
    print(f"Device:     {args.device}")
    print(f"Generation: {'enabled (final stage only)' if with_generation else 'disabled'}")
    print(f"Tasks:      {TASK_NAMES}")

    # Identify stages to evaluate
    if args.stage is not None:
        stages = [args.stage]
    else:
        stages = list(range(NUM_TASKS))

    # Verify checkpoints exist
    checkpoint_template = 'olora_multi_task_{stage}_best.pth'
    for s in stages:
        ckpt = exp_dir / checkpoint_template.format(stage=s)
        if not ckpt.exists():
            print(f"ERROR: checkpoint not found: {ckpt}")
            sys.exit(1)
    print(f"Stages:     {stages}")

    # Initialize word vectorizer
    print("\nLoading word vectorizer...")
    glove_path = str(PRETRAINED / 'glove')
    w_vectorizer = WordVectorizer(glove_path, 'our_vab')

    # Initialize NLGEval (only needed for generation)
    nlg_eval = None
    if with_generation and HAVE_NLGEVAL:
        disable_spice = str(os.environ.get("MOTION_AGENT_DISABLE_SPICE", "0")).strip().lower() in {"1", "true", "yes"}
        metrics_to_omit = [
            'METEOR',
            'EmbeddingAverageCosineSimilarity',
            'SkipThoughtCS',
            'VectorExtremaCosineSimilarity',
            'GreedyMatchingScore',
        ]
        if disable_spice:
            metrics_to_omit.append('SPICE')
            print("Loading NLGEval (BLEU, ROUGE-L, CIDEr; SPICE disabled via MOTION_AGENT_DISABLE_SPICE=1)...")
        else:
            print("Loading NLGEval (BLEU, ROUGE-L, CIDEr, SPICE)...")

        nlg_eval = NLGEval(metrics_to_omit=metrics_to_omit)
    elif with_generation and not HAVE_NLGEVAL:
        print("Warning: NLGEval unavailable — NLG metrics will be skipped.")

    # Run evaluation per stage
    all_results = {}
    for stage in stages:
        ckpt_path = exp_dir / checkpoint_template.format(stage=stage)

        # Default behaviour: generation only at final stage (stage 4)
        # Optional: enable generation at all stages for meaningful NLG BWT/FWT.
        with_generation_for_stage = with_generation and (
            args.generate_all_stages or (stage == NUM_TASKS - 1)
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
        all_results[f'after_task{stage}'] = stage_results

    # Print overall accuracy matrix
    print("\n" + "=" * 70)
    print("TOKEN ACCURACY MATRIX  R[stage, task]  (rows=stage, cols=task)")
    print("=" * 70)
    header = f"{'Stage':<12}" + "".join(f"{TASK_DISPLAY_NAMES.get(t, t)[:10]:>12}" for t in TASK_NAMES)
    print(header)
    print("-" * (12 + 12 * NUM_TASKS))
    for stage in stages:
        row = f"After T{stage}   "
        for task_name in TASK_NAMES:
            acc = all_results[f'after_task{stage}']['per_task'][task_name]['accuracy']
            row += f"{acc*100:>12.2f}"
        print(row)
    print("=" * 70)

    print("\nDone. Run compute_cl_metrics_m2t.py to compute BWT/FWT.")


if __name__ == '__main__':
    main()
