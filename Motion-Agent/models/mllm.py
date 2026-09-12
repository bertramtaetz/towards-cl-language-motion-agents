
from repo_paths import PRETRAINED, DATA_ROOT, TASK_SPLITS, BACKBONE, task_order
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch
from models.training_utils import *
import numpy as np
import models.vqvae as vqvae
from pathlib import Path

# Get the Motion-Agent root directory (parent of models/)
_MOTION_AGENT_ROOT = Path(__file__).parent.parent.resolve()

class MotionLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_backbone)
        self.llm = AutoModelForCausalLM.from_pretrained(self.args.llm_backbone)
        self.nb_text_tokens = len(self.tokenizer)
        self.mean = np.load(PRETRAINED / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
        self.std = np.load(PRETRAINED / 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')
        self.device = args.device
        self.training_task = None # t2m or m2t for training

        # LoRA target modules are configurable so baselines can be made
        # comparable to O-LoRA (q_proj/v_proj only) while keeping the default
        # Motion-Agent configuration unchanged.
        #
        # Supported modes:
        #   - "full" (default): 7 modules as in the original Motion-Agent code
        #   - "qv":             only q_proj and v_proj (as used by O-LoRA)
        def _resolve_target_modules(mode: str):
            mode = (mode or "full").lower()
            if mode in {"qv", "q_proj,v_proj", "q_proj_v_proj"}:
                return ["q_proj", "v_proj"]
            if mode in {"full", "default"}:
                return [
                    "o_proj", "q_proj", "up_proj", "v_proj",
                    "k_proj", "down_proj", "gate_proj",
                ]
            # Also allow explicit comma-separated lists.
            if "," in mode:
                return [m.strip() for m in mode.split(",") if m.strip()]
            raise ValueError(
                f"Unknown LoRA target-modules mode: {mode}. Use 'full', 'qv', or comma-separated module names."
            )

        t2m_targets = _resolve_target_modules(getattr(self.args, "lora_target_modules_t2m", "full"))
        m2t_targets = _resolve_target_modules(getattr(self.args, "lora_target_modules_m2t", "full"))

        self.lora_config_t2m = LoraConfig(
            r=self.args.lora_r_t2m,
            lora_alpha=self.args.lora_alpha_t2m,
            target_modules=t2m_targets,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.lora_config_m2t = LoraConfig(
            r=self.args.lora_r_m2t,
            lora_alpha=self.args.lora_alpha_m2t,
            target_modules=m2t_targets,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, self.lora_config_t2m, adapter_name='t2m')
        self.llm.add_adapter('m2t', self.lora_config_m2t)

        self.args.nb_joints = 22
        self.args.dataname = 't2m'
        self.args.vq_path = str(PRETRAINED / 'ckpt/vqvae.pth')
        self.net = vqvae.HumanVQVAE(self.args, ## use args to define different parameters in different quantizers
                           self.args.nb_code,
                           self.args.code_dim,
                           self.args.output_emb_width,
                           self.args.down_t,
                           self.args.stride_t,
                           self.args.width,
                           self.args.depth,
                           self.args.dilation_growth_rate,
                           self.args.vq_act,
                           self.args.vq_norm)
        print ('loading vqvae from {}'.format(self.args.vq_path))
        ckpt = torch.load(self.args.vq_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net.to(self.device)

        self.tokenizer.add_tokens(['<Motion>', '</Motion>'])
        self.motion_token_indices = np.arange(self.args.nb_code) 
        self.motion_token_indices = len(self.tokenizer) + self.motion_token_indices
        for i in range(self.args.nb_code):
            self.tokenizer.add_tokens([f'<Motion_{i}>'])
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # PEFT/LoRA freezes all base model params including embeddings and
        # lm_head. Keep them frozen: embeddings and lm_head are pretrained
        # representations that should not change during fine-tuning. Only the
        # LoRA adapter parameters should be trained -- this ensures a fair
        # apples-to-apples comparison across continual learning methods.

        self.llm.to(self.device)
        self.llm.eval()

        # print(self.llm)
    
    def forward(self, caption, motion):
        # Use override adapter if set (for continual learning with custom adapters)
        if hasattr(self, 'adapter_override') and self.adapter_override is not None:
            self.llm.set_adapter(self.adapter_override)
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')

        inputs_ids, targets, attention_mask = process_batch(tokenizer=self.tokenizer, 
                                                            batch_of_captions=caption, 
                                                            max_tgt_len=200, 
                                                            batch_of_motions=motion,
                                                            training_task=self.training_task)
        
        # print(inputs_ids.shape)
        # print(targets.shape)
        # print(attention_mask.shape)
        # print(tokenizer.decode(inputs_ids[0]))
        inputs_ids = inputs_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        targets = targets.to(self.device)

        outputs = self.llm(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            return_dict=True,
            # IMPORTANT: Do NOT request hidden states during training.
            # Keeping `output_hidden_states=True` retains activations for every
            # layer and can easily trigger OOM on large decoder-only LMs.
            labels=targets,
        )

        loss = outputs.loss
        # print(outputs.logits.shape)
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc, chosen_tokens, labels
    
    def generate(self, caption, verbose=False):
        """
        Generate motion tokens from a single caption.
        
        This method wraps generate_batch() for backward compatibility.
        For processing multiple captions, use generate_batch() directly.
        
        Args:
            caption: Text description of the motion
            verbose: Print timing information
        
        Returns:
            Motion token tensor
        """
        results = self.generate_batch([caption], max_length=200, verbose=verbose)
        return results[0] if results else torch.zeros(1, dtype=torch.long, device=self.device)
    
    def generate_batch(self, captions, max_length=200, verbose=False):
        """
        Batch generation of motion tokens from multiple captions.
        
        This method processes multiple captions in parallel, providing significant
        speedup over sequential calls to generate().
        
        Args:
            captions: List of text descriptions
            max_length: Maximum generation length (default: 200)
            verbose: Print timing information
        
        Returns:
            List of motion token tensors (variable length per sample)
        """
        import time as _time
        t0 = _time.time()
        
        if len(captions) == 0:
            return []
        
        # Use adapter_override if set (for continual learning evaluation)
        # Otherwise default to t2m for standard motion generation
        if hasattr(self, 'adapter_override') and self.adapter_override is not None:
            self.llm.set_adapter(self.adapter_override)
        else:
            self.llm.set_adapter('t2m')
        self.llm.eval()
        
        # Build prompts for all captions
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        
        batch_inputs = []
        for caption in captions:
            input_text = '### Input:\n' + caption + '\n\nResponse: <Motion>'
            full_input = prompt + instruction + input_text
            batch_inputs.append(full_input)
        
        # Tokenize with left padding (required for decoder-only batch generation)
        # Save original padding side and restore after
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Batch tokenize
        encoded = self.tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side
        
        t1 = _time.time()
        if verbose:
            print(f"[GEN_BATCH] Tokenization: {t1-t0:.2f}s, batch_size: {len(captions)}, input_shape: {input_ids.shape}")
        
        # Get the </Motion> token ID for EOS detection
        # </Motion> is at index 1 in the motion token space (after <Motion> at index 0)
        end_motion_token_id = self.nb_text_tokens + 1  # </Motion> token
        
        # Generate with batch
        outputs = self.llm.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=1,  # Greedy decoding for speed
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        t2 = _time.time()
        if verbose:
            print(f"[GEN_BATCH] LLM generate: {t2-t1:.2f}s, output_shape: {outputs.sequences.shape}")
        
        # Parse outputs for each sample
        batch_size = len(captions)
        results = []
        
        # Stack scores: [num_generated_tokens, batch_size, vocab_size]
        if len(outputs.scores) > 0:
            scores = torch.stack(outputs.scores)  # [gen_len, batch_size, vocab_size]
            
            for b in range(batch_size):
                # Get scores for this sample
                sample_scores = scores[:, b, :]  # [gen_len, vocab_size]
                
                # Extract motion logits (last nb_code+2 tokens in vocab)
                motion_logits = sample_scores[:, -(self.args.nb_code + 2):]
                motion_tokens = torch.argmax(motion_logits, dim=-1)  # [gen_len]
                
                # Find end_of_motion token (index=1 in motion space)
                if 1 in motion_tokens:
                    end_idx = motion_tokens.tolist().index(1)
                    if end_idx == 0:
                        # Model immediately output end token
                        motion_tokens = motion_tokens[:1]
                    else:
                        motion_tokens = motion_tokens[:end_idx]
                
                # Adjust for special tokens (subtract 2 for <Motion> and </Motion>)
                motion_tokens = torch.clamp(motion_tokens - 2, min=0)
                results.append(motion_tokens)
        else:
            # No tokens generated - return minimal tokens
            for _ in range(batch_size):
                results.append(torch.zeros(1, dtype=torch.long, device=self.device))
        
        t3 = _time.time()
        if verbose:
            print(f"[GEN_BATCH] Post-processing: {t3-t2:.2f}s, total: {t3-t0:.2f}s")
        
        return results
    
    def caption(self, motion):
        self.llm.set_adapter('m2t')
        self.llm.eval()

        motion = self.normalize(motion)
        motion = torch.from_numpy(motion).float().to(self.device).unsqueeze(0)
        motion_tokens = self.net.encode(motion).squeeze(0)

        motion_tokens = motion_tokens + self.nb_text_tokens + 2 # reindex the motion tokens

        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
        input_text = '### Input:\n' + "<Motion>" + self.tokenizer.decode(motion_tokens) + '</Motion>' + '\n\nResponse: '
        input_texts = prompt + instruction + input_text
        input_ids = self.tokenizer.encode(input_texts, return_tensors="pt").to(self.device)
        pred = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2
        )
        pred = pred[0, len(input_ids[0]):]
        pred = self.tokenizer.decode(pred)
        caption = pred.split('<eos>')[0]

        return caption
    
    def merge_pretrained_adapter(self, pretrained_path, direction='t2m'):
        """Load pretrained checkpoint and merge the relevant LoRA adapter into
        base model weights, then re-wrap with a fresh LoRA for CL training.

        After this call the model has:
        - Base Gemma weights with pretrained motion LoRA baked in
        - Fresh LoRA adapter (zero contribution, ready for CL training)
        - Pretrained motion embeddings and lm_head (frozen)
        """
        self.load_model(pretrained_path)
        keep = direction
        self.llm.set_adapter(keep)
        self.llm = self.llm.merge_and_unload(adapter_names=[keep])
        lora_config = self.lora_config_t2m if direction == 't2m' else self.lora_config_m2t
        self.llm = get_peft_model(self.llm, lora_config, adapter_name=keep)
        self.llm.to(self.device)
        print(f"Merged pretrained {keep} adapter into base model, fresh {keep} LoRA added")

    def enable_motion_token_interface_training(self):
        """Enable training ONLY for motion-token embedding/head rows.

        Why: for a "minimal pretraining" stage starting from a general LLM,
        the motion token rows in the input embedding matrix and lm_head are
        randomly initialized. If they remain frozen, the model cannot learn
        to produce/consume motion tokens.

        Implementation detail:
        - We set requires_grad=True on the full embedding/head weights.
        - We register gradient hooks that zero-out gradients for all rows
          corresponding to text tokens (< nb_text_tokens) so that ONLY the
          motion-token rows are updated.

        This avoids accidental full-vocabulary fine-tuning.
        """
        emb = self.llm.get_input_embeddings().weight
        head = self.llm.lm_head.weight

        emb.requires_grad = True
        head.requires_grad = True

        # Register hooks only once.
        if getattr(self, "_motion_interface_hooks", None) is None:
            self._motion_interface_hooks = []

        if getattr(self, "_motion_interface_enabled", False):
            return

        # Create masks (1 for motion rows, 0 for text rows).
        device = emb.device
        emb_mask = torch.zeros_like(emb, device=device)
        emb_mask[self.nb_text_tokens:] = 1
        head_mask = torch.zeros_like(head, device=device)
        head_mask[self.nb_text_tokens:] = 1

        self._motion_interface_hooks.append(emb.register_hook(lambda g: g * emb_mask))
        self._motion_interface_hooks.append(head.register_hook(lambda g: g * head_mask))
        self._motion_interface_enabled = True

    def disable_motion_token_interface_training(self):
        """Remove motion-interface gradient hooks (if present)."""
        hooks = getattr(self, "_motion_interface_hooks", None) or []
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._motion_interface_hooks = []
        self._motion_interface_enabled = False

    def save_model(self, path):
        # only save the lora weights of the model
        save_dict = {}
        for name, param in self.llm.named_parameters():
            if 'lora' in name:
                save_dict[name] = param

        # save the additional token embeddings
        embeddings = self.llm.get_input_embeddings().weight[self.nb_text_tokens:]
        save_dict['embeddings'] = embeddings

        # save the lm_head of the additional tokens
        lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]
        save_dict['lm_head'] = lm_head

        torch.save(save_dict, path)

    def load_model(self, path):
        print(f"Loading model from {path}")
        save_dict = torch.load(path, map_location=self.device)
        for name, param in self.llm.named_parameters():
            if name in save_dict:
                param.data = save_dict[name]
        self.llm.get_input_embeddings().weight.data[self.nb_text_tokens:] = save_dict['embeddings']
        self.llm.lm_head.weight.data[self.nb_text_tokens:] = save_dict['lm_head']

    def denormalize(self, motion):
        return self.mean + motion * self.std

    def normalize(self, motion):
        return (motion - self.mean) / self.std
    
