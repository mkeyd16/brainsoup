"""FMSP -- Fine-tuning with Minimal Parameter changes for Small-parameter LMs.

FMSP is a fine-tuning technique purpose-built for **small** language models
(< 1M parameters). At this scale, every parameter carries useful information:
unlike large models fine-tuned with LoRA (which assume redundant capacity that
can be compressed into low-rank updates), a small model has *no* spare
dimensions. Full fine-tuning on a small task dataset causes severe catastrophic
forgetting; fine-tuning too few parameters has no effect.

FMSP resolves this tension with three coordinated mechanisms:

  1. **Fisher Information Profiling (FIP)** -- Before fine-tuning, the
     pretrained model's per-parameter Fisher information is estimated on a
     held-out sample of the *large* (pretraining) dataset. The diagonal
     Fisher information

         F_i = E[( d log p(x|theta) / d theta_i )^2]

     measures how much each parameter contributes to the model's loss
     landscape on language data. High-Fisher parameters are *critical* for
     language ability; low-Fisher parameters are safer to adapt.

  2. **Per-Parameter Gradient Masking (PGM)** -- Rather than coarse
     per-tensor freezing (``requires_grad=False``), FMSP registers a
     gradient hook on every parameter tensor that zeroes out the gradient
     at the *per-scalar* level for high-Fisher parameters. This gives
     single-weight granularity: exactly the bottom ``1 - freeze_fraction``
     of parameters (by Fisher score) are updated, and the rest are frozen
     in place. This is the "minimal parameter changes" guarantee.

  3. **Task Adapter Injection + KL Preservation (TAI+KPR)** -- A tiny
     rank-r adapter (default r=4) is injected after the model's final
     normalisation layer via a forward hook. The adapter's up-projection
     is zero-initialised, so the model starts bit-identical to the
     pretrained baseline. During fine-tuning, a KL-divergence penalty
     between the fine-tuned model's output distribution and a frozen
     reference (pretrained) model's output distribution explicitly anchors
     the model near its language ability:

         L = CE_qa + lambda * KL( p_ft || p_pretrained )

== Why FMSP is NOT LoRA ==

  - LoRA applies uniform low-rank updates to *every* linear layer; FMSP
    selectively updates only the lowest-Fisher parameters (automatic,
    data-driven selection, not architectural heuristic).
  - LoRA assumes the base model has redundant capacity to compress; FMSP
    is designed for models where every parameter matters.
  - LoRA has no explicit anti-forgetting mechanism; FMSP uses an explicit
    KL preservation loss against a frozen reference.
  - LoRA is representation-agnostic; FMSP's Fisher profiling is computed
    on the LM's own cross-entropy objective, making it language-model-
    optimised by construction.

== Philosophy compliance ==

FMSP does not change the backbone architecture. It adds at most
``2 * d_model * rank`` new parameters (the adapter), which is < 0.5% of
any preset. The backbone remains pure MLP-Mixer with no attention.

== Usage ==

    from src.fmsp import FMSPConfig, run_fmsp_finetune

    # 1. Pretrain the model on a large dataset (use existing train_vXX.py).
    # 2. Load the pretrained checkpoint.
    # 3. Run FMSP fine-tuning on a small task dataset:

    fmsp_cfg = FMSPConfig(
        freeze_fraction=0.75,      # freeze top-75% highest-Fisher params
        adapter_rank=4,            # tiny adapter
        kl_weight=0.1,             # preservation strength
        fisher_num_samples=500,
        learning_rate=3e-4,        # 10x lower than pretrain (3e-3)
        max_epochs=5,
    )
    metrics = run_fmsp_finetune(
        model=model,               # freshly loaded pretrained model
        pretrain_loader=pretrain_loader,  # large dataset (for Fisher)
        finetune_loader=qa_loader,        # small QA dataset
        val_loader=qa_val_loader,
        config=fmsp_cfg,
        device=device,
        checkpoint_dir="checkpoints/v46-fmsp/",
        model_name="V46-1",
    )
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FMSPConfig:
    """Configuration for FMSP fine-tuning.

    Fields are grouped:
      1. Fisher profiling.
      2. Parameter selection (freeze mask).
      3. Adapter injection.
      4. KL preservation loss.
      5. Optimisation.
    """

    # ---- Fisher profiling ------------------------------------------------
    fisher_num_samples: int = 500       # samples from pretrain data for Fisher
    fisher_batch_size: int = 4          # micro-batch for Fisher (low memory)

    # ---- parameter selection ---------------------------------------------
    freeze_fraction: float = 0.75       # fraction of params to FREEZE (highest Fisher)
    # 0.0 = freeze nothing (full fine-tune); 1.0 = freeze everything.
    true_freeze: bool = True            # restore-after-step freeze (V77-C2)
    # The grad-hook freeze only zeroes gradients; torch.optim.AdamW applies
    # DECOUPLED weight decay (lr * wd per step) to every parameter regardless
    # of its gradient, so "frozen" scalars silently shrink toward zero
    # (measured: 4.69% uniform shrink of 49,663 frozen scalars in one run).
    # When True, frozen scalars are snapshotted after apply_freeze_mask and
    # bit-exactly restored to their pre-FMSP values after every optimizer
    # step (p.data[mask] = snapshot). Optimizer-agnostic; empty masks are
    # a no-op. When False, preserves the original (leaky) behavior.

    # ---- parameter zoning (V78) ------------------------------------------
    zone_mode: str = "off"              # "off" = canonical FMSP; "hybrid" = zoned
    # In "hybrid" mode the freeze mask is computed as usual (top
    # freeze_fraction Fisher scalars frozen) and then every parameter whose
    # name contains any zone_structural_include substring is FORCED
    # trainable, regardless of its Fisher score. Trainable set =
    # Fisher bottom (1 - freeze_fraction) UNION structural includes.
    # Structural-include scalars are therefore never snapshotted/restored
    # by true_freeze (only the frozen set is). "off" is bit-identical to
    # canonical behavior.
    zone_structural_include: tuple = ("remixer.", "fmsp_adapter.")

    # ---- objective (V77-C1) ----------------------------------------------
    answer_only_ce: bool = False        # restrict CE to answer-region targets
    # Objective change: CE is computed ONLY over answer-region target
    # positions (a_mask shifted into target space: loss at position t is
    # kept iff a_mask[t+1]); question bytes are no longer trained, so the
    # QA format is not learned as an unconditional memorized-answer trigger.
    # The KL term is UNCHANGED -- still full non-pad sequence on the QA
    # batch. Validation (_fmsp_validate) applies the same masked CE so
    # checkpoint selection uses the training objective.

    # ---- adapter injection -----------------------------------------------
    use_adapter: bool = True            # inject a rank-r adapter
    adapter_rank: int = 4               # bottleneck dimension
    adapter_dropout: float = 0.0        # dropout in adapter (0 = off)

    # ---- KL preservation loss --------------------------------------------
    kl_weight: float = 0.1             # lambda: weight on KL(p_ft || p_ref)
    kl_temperature: float = 1.0        # temperature for KL softmax (1.0 = sharp)

    # ---- probe-KL: off-distribution conservation (V77 Stage 3) ------------
    probe_kl_weight: float = 0.0       # lambda2: weight on probe-batch KL; 0 disables
    # Probe batches are inputs the model was NOT taught on (Discord register
    # anchors, unseen 1k questions, weird-OOD prompts). The target is the
    # frozen backbone's output distribution: "don't move where you weren't
    # taught". NO cross-entropy is computed on probes -- this is not replay;
    # zero capacity is spent learning probe content. The KL is computed over
    # ALL non-pad positions of the full probe sequence (same
    # kl_preservation_loss as the QA KL). lambda2 is constant (no ramp).
    # Requires probe_dataloader to be passed to run_fmsp_finetune.
    probe_kl_temperature: float = 1.0  # temperature for probe-KL softmax

    # ---- probe margin/entropy-floor loss (V77 Stage 5, "M") ---------------
    probe_margin_weight: float = 0.0   # lambda3: weight on the margin loss; 0 disables
    # Escalation of probe-KL: probe-KL passively anchors the model to the
    # backbone on off-distribution inputs, but the "any `Assistant:` slot ->
    # confident memorized answer" association can survive it (measured:
    # fabrication 16.7/18 at probe_kl_weight=0.5). The margin loss ACTIVELY
    # floors the answer-start entropy on cue-ended probes: at positions where
    # the probe batch's "margin_mask" is True, the model's next-token
    # distribution must be AT LEAST as uncertain as the backbone's:
    #
    #     L_margin = mean_{t in margin_mask} relu(H_target - H(p_t))
    #
    # with H(p_t) = -(p * log p).sum(-1) from the ft probe logits (softmax
    # T=1.0). Trained questions are unaffected -- the mask is only True at
    # probe answer-start positions, so trained-question sharpness (kept by
    # answer-only CE) is preserved. If the margin is enabled but a probe
    # batch lacks "margin_mask", the batch contributes zero margin loss and a
    # one-time warning is printed (tolerated, not fatal).
    probe_margin_entropy: float = 4.0  # H_target in nats; override with the
    # measured backbone answer-start entropy (diag_v77_answerstart_entropy.py;
    # V76-RPG backbone measures 3.59 nats on the 800 probe 1k questions).

    # ---- optimisation ----------------------------------------------------
    learning_rate: float = 3e-4        # fine-tune LR (10x lower than pretrain 3e-3)
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1        # fraction of total steps for warmup
    max_grad_norm: float = 1.0
    max_epochs: int = 5
    val_split: float = 0.10
    log_interval: int = 25
    seed: int = 42

    # ---- scheduled sampling -----------------------------------------------
    ss_max_prob: float = 0.0       # 0=off; 0.2-0.5 typical
    ss_warmup_frac: float = 0.3   # epochs before SS starts (fraction of total)
    ss_answer_only: bool = True

    # ---- question-region byte dropout --------------------------------------
    q_dropout_prob: float = 0.0    # 0=off; replaces q_mask-region input bytes
                                   # with random bytes (labels untouched) to
                                   # prevent memorising surface question forms

    # ---- output ----------------------------------------------------------
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# 1. Fisher Information Profiling (FIP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _move_batch_to_device(batch, device):
    """Move a batch dict/tuple to device. Returns (input_ids, labels, q_mask, a_mask, all_mask)."""
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", batch.get("targets"))
        if labels is not None:
            labels = labels.to(device)
        q_mask = batch.get("q_mask")
        a_mask = batch.get("a_mask")
        all_mask = batch.get("all_mask")
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device) if labels is not None else None
        q_mask = a_mask = all_mask = None
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    if q_mask is not None:
        q_mask = q_mask.to(device)
    if a_mask is not None:
        a_mask = a_mask.to(device)
    if all_mask is not None:
        all_mask = all_mask.to(device)
    return input_ids, labels, q_mask, a_mask, all_mask


def compute_fisher_information(
    model: nn.Module,
    dataloader,
    num_samples: int = 500,
    device: str = "cuda",
    max_seq_len: int = 512,
) -> dict[str, torch.Tensor]:
    """Compute the diagonal Fisher information for each parameter.

    The Fisher information is estimated as the mean squared gradient of the
    cross-entropy loss over ``num_samples`` batches from the *pretraining*
    data:

        F_i ~= (1/N) * sum_n ( d CE_n / d theta_i )^2

    High F_i means parameter theta_i is important for the model's language
    ability (small perturbations cause large loss increases). These are the
    parameters FMSP should NOT touch during fine-tuning.

    Args:
        model: pretrained model (will be put in train mode for gradient
            computation, then restored).
        dataloader: yields batches from the large pretraining dataset.
        num_samples: number of batches to accumulate over.
        device: compute device.
        max_seq_len: unused (kept for API stability); batches are used as-is.

    Returns:
        Dict mapping parameter name -> Fisher score tensor (same shape as
        the parameter). All values are non-negative.
    """
    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.train()

    fisher: dict[str, torch.Tensor] = {
        name: torch.zeros_like(p.data, device=p.device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break

        input_ids, labels, q_mask, a_mask, all_mask = _move_batch_to_device(batch, device)
        if labels is None:
            labels = input_ids.clone()

        model.zero_grad(set_to_none=True)
        try:
            logits, loss, _ = model(
                input_ids=input_ids, q_mask=q_mask, a_mask=a_mask,
                all_mask=all_mask, targets=labels,
            )
        except TypeError:
            # Fallback for models that don't accept all keyword args.
            loss = model(input_ids)

        if not torch.isfinite(loss):
            continue

        loss.backward()

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None and name in fisher:
                    fisher[name] += p.grad.detach().pow(2)
        count += 1

    if count == 0:
        raise RuntimeError(
            "Fisher profiling processed 0 batches. Check the dataloader."
        )

    for name in fisher:
        fisher[name] /= count

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()

    return fisher


# ---------------------------------------------------------------------------
# 2. Per-Parameter Gradient Masking (PGM)
# ---------------------------------------------------------------------------

def build_freeze_mask(
    fisher: dict[str, torch.Tensor],
    freeze_fraction: float,
) -> dict[str, torch.Tensor]:
    """Build a per-scalar freeze mask from Fisher scores.

    Freezes exactly the top ``freeze_fraction`` of scalars by Fisher score
    (highest Fisher = most important for language = frozen). Uses ``topk``
    for exact fraction control, avoiding quantile ties when many params
    share the same Fisher score (e.g. unactivated bucket embeddings have
    Fisher = 0).

    Args:
        fisher: output of ``compute_fisher_information``.
        freeze_fraction: fraction of scalars to freeze (0.0 to 1.0).
            0.75 = freeze the highest-Fisher 75% of scalars.

    Returns:
        Dict {param_name: bool tensor}, same shape as each parameter.
        True = frozen (do not update), False = trainable.
    """
    if not (0.0 <= freeze_fraction <= 1.0):
        raise ValueError(f"freeze_fraction must be in [0, 1], got {freeze_fraction}")

    names = list(fisher.keys())
    flat_scores = torch.cat([fisher[n].flatten().to(torch.float32) for n in names])
    n_total = flat_scores.numel()
    n_freeze = int(round(n_total * freeze_fraction))

    if n_freeze == 0:
        return {n: torch.zeros_like(f, dtype=torch.bool) for n, f in fisher.items()}
    if n_freeze >= n_total:
        return {n: torch.ones_like(f, dtype=torch.bool) for n, f in fisher.items()}

    _, top_idx = torch.topk(flat_scores, n_freeze)
    global_freeze = torch.zeros(n_total, dtype=torch.bool)
    global_freeze[top_idx] = True

    freeze_mask: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        size = fisher[name].numel()
        freeze_mask[name] = global_freeze[offset:offset + size].reshape(fisher[name].shape)
        offset += size
    return freeze_mask


def _force_zone_includes(
    freeze_mask: dict[str, torch.Tensor],
    include_patterns: tuple,
) -> int:
    """V78 hybrid zoning: force include-pattern params trainable (mask=False).

    Substring match on parameter names. Returns the number of scalars that
    were frozen by Fisher rank and are now forced into the trainable zone.
    """
    n_forced = 0
    for name in list(freeze_mask.keys()):
        if any(pat in name for pat in include_patterns):
            n_forced += int(freeze_mask[name].sum().item())
            freeze_mask[name] = torch.zeros_like(freeze_mask[name])
    return n_forced


def apply_freeze_mask(
    model: nn.Module,
    freeze_mask: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Apply per-scalar gradient masking via backward hooks.

    For each parameter with a freeze mask, a backward hook is registered
    that zeroes out the gradient at frozen positions. This achieves
    per-scalar freeze granularity (finer than PyTorch's per-tensor
    ``requires_grad=False``).

    The trainable-parameter count (scars with grad) is unchanged -- only
    the gradient values are masked. This keeps the optimizer happy (no
    need to rebuild param groups).

    Args:
        model: the model to mask.
        freeze_mask: output of ``build_freeze_mask``.

    Returns:
        Summary dict: {total_scalars, frozen_scalars, trainable_scalars,
        frozen_fraction}.
    """
    total = 0
    frozen = 0
    handles = []
    model._fmsp_freeze_handles = handles  # store for later removal

    for name, p in model.named_parameters():
        if name not in freeze_mask:
            continue
        mask = freeze_mask[name]
        if mask.device != p.device:
            mask = mask.to(p.device)
        # The "keep" mask: 1.0 where trainable, 0.0 where frozen.
        keep = (~mask).to(p.dtype)

        total += mask.numel()
        frozen += int(mask.sum().item())

        if not mask.any():
            continue  # nothing frozen in this tensor

        def make_hook(keep_mask):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                return grad * keep_mask
            return hook

        h = p.register_hook(make_hook(keep))
        handles.append(h)

    trainable = total - frozen
    return {
        "total_scalars": total,
        "frozen_scalars": frozen,
        "trainable_scalars": trainable,
        "frozen_fraction": frozen / max(total, 1),
    }


def remove_freeze_mask(model: nn.Module) -> None:
    """Remove all FMSP gradient-masking hooks from the model."""
    handles = getattr(model, "_fmsp_freeze_handles", [])
    for h in handles:
        h.remove()
    model._fmsp_freeze_handles = []


# ---------------------------------------------------------------------------
# 3. Task Adapter Injection (TAI)
# ---------------------------------------------------------------------------

class FMSPAdapter(nn.Module):
    """Rank-r bottleneck adapter for task-specific adaptation.

    Structure: ``Linear(d, r) -> GELU -> Linear(r, d)`` with the
    up-projection zero-initialised. At init the adapter output is exactly
    zero, so the model behaves identically to its pretrained baseline.
    During fine-tuning, the adapter learns a small task-specific residual.

    The adapter is attached AFTER the model's final normalisation layer
    (``out_norm``) via a forward hook, so it modifies the representation
    just before the tied output head. This is the single most impactful
    injection point for steering the model's output distribution.

    Parameter count: ``2 * d_model * rank`` (no bias). For d=80, r=4: 640
    params (< 0.07% of V45's 956K).
    """

    def __init__(self, d_model: int, rank: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.down = nn.Linear(d_model, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialise: down = Kaiming, up = zero (identity at init).
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) -> (..., d_model) residual (zero at init)."""
        return self.dropout(self.up(self.act(self.down(h))))


def attach_adapter(
    model: nn.Module,
    d_model: int,
    rank: int = 4,
    dropout: float = 0.0,
    target_layer_name: str = "out_norm",
) -> tuple[FMSPAdapter, object]:
    """Attach an FMSP adapter to a model via a forward hook.

    The adapter is registered as ``model.fmsp_adapter`` (making it a
    submodule whose parameters appear in ``state_dict``). A forward hook
    on ``model.<target_layer_name>`` adds the adapter's output to the
    layer's output:

        out_norm_output -> out_norm_output + adapter(out_norm_output)

    Because the adapter is zero-init, the model is bit-identical to its
    pretrained baseline at attach time.

    Args:
        model: the model to attach to.
        d_model: hidden dimension.
        rank: adapter bottleneck dimension.
        dropout: adapter dropout rate.
        target_layer_name: name of the submodule to hook (default
            ``"out_norm"``; all MicroMixer models expose this).

    Returns:
        (adapter, hook_handle). Save the handle to remove the hook later
        if needed. The adapter is also accessible as ``model.fmsp_adapter``.
    """
    adapter = FMSPAdapter(d_model, rank=rank, dropout=dropout)
    model.fmsp_adapter = adapter  # registers as submodule

    target_layer = getattr(model, target_layer_name)
    if target_layer is None:
        raise AttributeError(
            f"Model has no attribute '{target_layer_name}'. "
            f"FMSP requires a normalisation layer to hook the adapter onto."
        )

    def hook(module, inp, out):
        # Adapter is applied in the model's current mode (train/eval).
        return out + adapter(out)

    handle = target_layer.register_forward_hook(hook)
    model._fmsp_adapter_handle = handle
    return adapter, handle


def detach_adapter(model: nn.Module) -> None:
    """Remove the FMSP adapter and its hook from the model."""
    handle = getattr(model, "_fmsp_adapter_handle", None)
    if handle is not None:
        handle.remove()
        model._fmsp_adapter_handle = None
    if hasattr(model, "fmsp_adapter"):
        del model.fmsp_adapter


# ---------------------------------------------------------------------------
# 4. KL Preservation Loss (KPR)
# ---------------------------------------------------------------------------

def kl_preservation_loss(
    logits_ft: torch.Tensor,
    logits_ref: torch.Tensor,
    non_pad_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(p_ft || p_ref) averaged over non-pad positions.

    Anchors the fine-tuned model's output distribution near the frozen
    pretrained model's distribution. This is the explicit anti-forgetting
    mechanism: the model can adapt to QA, but it pays a quadratic penalty
    for drifting too far from its language-ability distribution.

    The KL is computed per-position then averaged over non-pad positions:

        KL = mean_pos sum_v p_ft(v) * (log p_ft(v) - log p_ref(v))

    where p = softmax(logits / T).

    Args:
        logits_ft: (B, S, V) fine-tuned model logits (with adapter + updated weights).
        logits_ref: (B, S, V) reference model logits (frozen pretrained).
        non_pad_mask: (B, S) bool, True at non-pad positions.
        temperature: softmax temperature (higher = softer distributions).

    Returns:
        Scalar KL divergence (averaged over non-pad positions).
    """
    # Mask: (B, S) -> (B, S, 1) for broadcasting over vocab.
    mask = non_pad_mask.unsqueeze(-1).to(logits_ft.dtype)

    log_p_ft = F.log_softmax(logits_ft / temperature, dim=-1)
    p_ft = log_p_ft.exp()
    log_p_ref = F.log_softmax(logits_ref / temperature, dim=-1)

    # KL(p_ft || p_ref) = sum p_ft * (log p_ft - log p_ref)
    kl_per_pos = (p_ft * (log_p_ft - log_p_ref)).sum(dim=-1)  # (B, S)
    kl_masked = kl_per_pos * non_pad_mask.to(kl_per_pos.dtype)
    denom = non_pad_mask.to(kl_per_pos.dtype).sum().clamp(min=1)
    return kl_masked.sum() / denom


# ---------------------------------------------------------------------------
# 5. FMSP fine-tuning loop
# ---------------------------------------------------------------------------

def _answer_masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    a_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Cross-entropy restricted to answer-region TARGET positions (V77-C1).

    Shift semantics: labels[t] is the target for position t and equals
    input_ids[t+1]; a_mask marks answer-region positions in INPUT space.
    The loss at position t is therefore gated by whether the TARGET byte is
    an answer byte: loss_mask[t] = a_mask[t+1].

    Args:
        logits: (B, S, V) model logits.
        labels: (B, S) next-token targets (labels[t] = input_ids[t+1]).
        a_mask: (B, S) bool, answer-region positions in input space.
        pad_token_id: ignored target id.

    Returns:
        Scalar CE averaged over answer-region, non-pad target positions.
    """
    V = logits.shape[-1]
    ce_per_pos = F.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1),
        ignore_index=pad_token_id, reduction="none",
    ).reshape(labels.shape)
    loss_mask = torch.zeros_like(a_mask, dtype=ce_per_pos.dtype)
    loss_mask[:, :-1] = a_mask[:, 1:].to(ce_per_pos.dtype)
    loss_mask[:, -1] = 0
    loss_mask = loss_mask * (labels != pad_token_id)
    return (ce_per_pos * loss_mask).sum() / loss_mask.sum().clamp(min=1)


def probe_margin_loss(
    logits_ft: torch.Tensor,
    margin_mask: torch.Tensor,
    h_target: float,
) -> torch.Tensor:
    """Entropy-floor margin loss at masked positions (V77 Stage 5, "M").

    H(p_t) = -(p * log p).sum(-1) with p = softmax(logits_ft) (T=1.0);
    L = mean over masked positions of relu(h_target - H(p_t)). Zero when the
    model is already at least as uncertain as the target everywhere masked.
    """
    log_p = F.log_softmax(logits_ft.float(), dim=-1)
    p = log_p.exp()
    entropy = -(p * log_p).sum(dim=-1)  # (B, S), nats
    deficit = F.relu(h_target - entropy)
    mask = margin_mask.to(deficit.dtype)
    denom = mask.sum().clamp(min=1)
    return (deficit * mask).sum() / denom


def _prepare_batch(batch, device):
    """Extract and move batch to device. Mirrors Trainer._prepare_batch."""
    return _move_batch_to_device(batch, device)


def _make_wsd_schedule(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    stable_frac: float = 0.7,
):
    """Build a WSD (Warmup-Stable-Decay) LR schedule.

    Phase 1 (warmup): linear ramp 0 -> base_lr over warmup_steps.
    Phase 2 (stable): constant base_lr.
    Phase 3 (decay): linear decay base_lr -> 0 over the final (1-stable_frac).
    """
    stable_steps = int(total_steps * stable_frac)
    decay_steps = total_steps - warmup_steps - stable_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < warmup_steps + stable_steps:
            return 1.0
        # Decay phase.
        decay_step = step - warmup_steps - stable_steps
        return max(0.0, 1.0 - decay_step / max(decay_steps, 1))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_fmsp_finetune(
    model: nn.Module,
    pretrain_dataloader,
    finetune_dataloader,
    val_dataloader,
    config: FMSPConfig,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints/fmsp",
    model_name: str = "V46",
    d_model: int | None = None,
    verbose: bool = True,
    probe_dataloader=None,
) -> dict:
    """End-to-end FMSP fine-tuning.

    Pipeline:
      0. Snapshot a frozen reference model (for KL preservation).
      1. Compute Fisher information on the pretrain data.
      2. Build + apply per-scalar freeze mask.
      3. Attach the zero-init adapter.
      4. Fine-tune with CE + lambda * KL on the small task dataset
         (+ lambda2 * probe-KL on off-distribution probe batches when
         config.probe_kl_weight > 0).
      5. Save checkpoints + report metrics.

    Args:
        model: freshly-loaded pretrained model (will be modified in place:
            freeze mask hooks + adapter are attached).
        pretrain_dataloader: large dataset (for Fisher profiling only).
        finetune_dataloader: small task dataset (for fine-tuning).
        val_dataloader: validation split of the task dataset.
        config: FMSPConfig.
        device: compute device.
        checkpoint_dir: where to save checkpoints.
        model_name: display name for logging.
        d_model: hidden dimension (for adapter). If None, inferred from
            the model's out_norm or config.
        verbose: print progress.
        probe_dataloader: off-distribution probe batches (V77 Stage 3).
            Required when config.probe_kl_weight > 0 or
            config.probe_margin_weight > 0; ignored otherwise.
            Batches are prepare_qaa_batch-style dicts; ONLY input_ids and
            the non-pad mask are used. No CE is ever computed on probes.
            When a batch carries a "margin_mask" bool tensor and
            config.probe_margin_weight > 0, the entropy-floor margin loss
            is applied at the masked (answer-start) positions.

    Returns:
        Metrics dict with keys: fisher_summary, trainable_scalars,
        train_loss_history, val_loss_history, final_val_loss.
    """
    if (config.probe_kl_weight > 0 or config.probe_margin_weight > 0) \
            and probe_dataloader is None:
        raise ValueError(
            "config.probe_kl_weight > 0 or config.probe_margin_weight > 0 "
            "requires probe_dataloader (off-distribution probe batches for "
            "the conservation KL / margin loss). "
            "Set probe_kl_weight=0.0 and probe_margin_weight=0.0 to disable."
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    # ---- 0. Snapshot frozen reference model -----------------------------
    if verbose:
        print(f"\n[{model_name}] FMSP Phase 0: snapshotting reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    model.to(device)

    # ---- Infer d_model if not given -------------------------------------
    if d_model is None:
        if hasattr(model, "config") and hasattr(model.config, "d_model"):
            d_model = model.config.d_model
        elif hasattr(model, "out_norm") and hasattr(model.out_norm, "weight"):
            d_model = model.out_norm.weight.shape[0]
        else:
            raise ValueError(
                "Could not infer d_model. Pass it explicitly via d_model=..."
            )

    # ---- 1. Fisher information profiling --------------------------------
    if verbose:
        print(f"[{model_name}] FMSP Phase 1: Fisher profiling "
              f"({config.fisher_num_samples} samples)...")
    fisher = compute_fisher_information(
        model, pretrain_dataloader,
        num_samples=config.fisher_num_samples,
        device=device,
    )

    # ---- 2. Build + apply freeze mask -----------------------------------
    if verbose:
        print(f"[{model_name}] FMSP Phase 2: building freeze mask "
              f"(freeze_fraction={config.freeze_fraction})...")
    freeze_mask = build_freeze_mask(fisher, config.freeze_fraction)
    if config.zone_mode == "hybrid":
        n_forced = _force_zone_includes(freeze_mask,
                                        config.zone_structural_include)
        if verbose:
            print(f"  zone_mode=hybrid: {n_forced:,} structural-include "
                  f"scalars forced trainable "
                  f"(patterns: {config.zone_structural_include})")
    elif config.zone_mode != "off":
        raise ValueError(
            f"zone_mode must be 'off' or 'hybrid', got {config.zone_mode!r}")
    freeze_summary = apply_freeze_mask(model, freeze_mask)
    if verbose:
        print(f"  Frozen: {freeze_summary['frozen_scalars']:,} / "
              f"{freeze_summary['total_scalars']:,} scalars "
              f"({freeze_summary['frozen_fraction']*100:.1f}%)")

    # Snapshot frozen scalars for restore-after-step (true freeze).
    # Built after apply_freeze_mask, before adapter attach / training, so
    # values come from the pretrained weights (post-load, pre-training).
    frozen_snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if config.true_freeze:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name not in freeze_mask:
                    continue
                mask = freeze_mask[name].to(p.device)
                if not mask.any():
                    continue  # nothing frozen in this tensor (e.g. freeze_fraction=0.0)
                frozen_snapshot[name] = (mask, p.data[mask].clone())
        if verbose:
            n_snap = sum(int(m.sum().item()) for m, _ in frozen_snapshot.values())
            print(f"  true_freeze: {n_snap:,} frozen scalars will be "
                  f"bit-exactly preserved (AdamW wd-leak blocked)")

    # ---- 3. Attach adapter ----------------------------------------------
    adapter_param_count = 0
    if config.use_adapter:
        if verbose:
            print(f"[{model_name}] FMSP Phase 3: attaching adapter "
                  f"(rank={config.adapter_rank})...")
        attach_adapter(
            model, d_model,
            rank=config.adapter_rank,
            dropout=config.adapter_dropout,
        )
        adapter_param_count = sum(
            p.numel() for p in model.fmsp_adapter.parameters()
        )
        model.fmsp_adapter.to(device)
        if verbose:
            print(f"  Adapter params: {adapter_param_count:,} "
                  f"({adapter_param_count / freeze_summary['total_scalars'] * 100:.3f}% of base)")

    # ---- 4. Fine-tune with CE + KL --------------------------------------
    if verbose:
        print(f"[{model_name}] FMSP Phase 4: fine-tuning "
              f"({config.max_epochs} epochs, lr={config.learning_rate}, "
              f"kl_weight={config.kl_weight})...")

    # Collect trainable params (all params, since freeze is via grad hooks).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(finetune_dataloader) * config.max_epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = _make_wsd_schedule(
        optimizer, warmup_steps, total_steps, config.learning_rate,
    )

    train_loss_history = []
    val_loss_history = []
    global_step = 0
    named_params = dict(model.named_parameters()) if frozen_snapshot else {}

    probe_iter = None
    if config.probe_kl_weight > 0 or config.probe_margin_weight > 0:
        try:
            n_probes = len(probe_dataloader.dataset)
        except (TypeError, AttributeError):
            n_probes = -1
        if verbose:
            mgl_str = (f", margin lambda3={config.probe_margin_weight} "
                       f"H_target={config.probe_margin_entropy}"
                       if config.probe_margin_weight > 0 else "")
            print(f"[{model_name}] Probe-KL enabled: {n_probes} probes, "
                  f"lambda2={config.probe_kl_weight}, "
                  f"T={config.probe_kl_temperature} (no CE on probes){mgl_str}")
        probe_iter = itertools.cycle(probe_dataloader)

    for epoch in range(config.max_epochs):
        model.train()
        epoch_start = time.time()
        running_ce = 0.0
        running_kl = 0.0
        running_pkl = 0.0
        running_mgl = 0.0
        running_acc = 0.0
        num_batches = 0

        if config.ss_max_prob > 0:
            warmup_epochs = max(1, int(config.max_epochs * config.ss_warmup_frac))
            if epoch < warmup_epochs:
                ss_prob = 0.0
            else:
                progress = (epoch - warmup_epochs) / max(1, config.max_epochs - warmup_epochs)
                ss_prob = config.ss_max_prob * progress
        else:
            ss_prob = 0.0

        for batch in finetune_dataloader:
            input_ids, labels, q_mask, a_mask, all_mask = _prepare_batch(batch, device)
            if labels is None:
                labels = input_ids.clone()

            ss_input_ids = input_ids
            if ss_prob > 0:
                with torch.no_grad():
                    ss_out = model(
                        input_ids=input_ids, q_mask=q_mask, a_mask=a_mask,
                        all_mask=all_mask, targets=None,
                    )
                    ss_logits = ss_out[0] if isinstance(ss_out, tuple) else ss_out
                    pred = ss_logits.argmax(dim=-1)

                pred_shifted = torch.cat([
                    torch.full((pred.shape[0], 1), config.pad_token_id,
                               dtype=pred.dtype, device=device),
                    pred[:, :-1],
                ], dim=1)

                rand = torch.rand(input_ids.shape, device=device)
                replace_mask = rand < ss_prob
                replace_mask[:, 0] = False
                if config.ss_answer_only and a_mask is not None:
                    replace_mask = replace_mask & a_mask
                non_pad_orig = (input_ids != config.pad_token_id)
                replace_mask = replace_mask & non_pad_orig

                ss_input_ids = torch.where(replace_mask, pred_shifted, input_ids)

            if config.q_dropout_prob > 0 and q_mask is not None:
                rand_q = torch.rand(input_ids.shape, device=device)
                drop = (rand_q < config.q_dropout_prob) & q_mask & (input_ids != config.pad_token_id)
                rand_bytes = torch.randint(3, 256, input_ids.shape, dtype=input_ids.dtype, device=device)
                ss_input_ids = torch.where(drop, rand_bytes, ss_input_ids)

            logits_ft, ce_loss, aux = model(
                input_ids=ss_input_ids, q_mask=q_mask, a_mask=a_mask,
                all_mask=all_mask, targets=labels,
            )

            # V77-C1: replace full-sequence CE with answer-region-only CE.
            ans_acc = None
            if config.answer_only_ce:
                if a_mask is not None:
                    ce_loss = _answer_masked_ce(
                        logits_ft, labels, a_mask, config.pad_token_id,
                    )
                    with torch.no_grad():
                        tgt_mask = torch.zeros_like(a_mask)
                        tgt_mask[:, :-1] = a_mask[:, 1:]
                        tgt_mask = tgt_mask & (labels != config.pad_token_id)
                        denom = tgt_mask.sum().clamp(min=1)
                        ans_acc = ((logits_ft.argmax(-1) == labels) & tgt_mask).float().sum() / denom
                elif not getattr(run_fmsp_finetune, "_ans_ce_warned", False):
                    print(f"  [{model_name}] WARNING: answer_only_ce=True but "
                          f"a_mask is None; falling back to full-sequence CE.")
                    run_fmsp_finetune._ans_ce_warned = True

            with torch.no_grad():
                ref_out = ref_model(
                    input_ids=ss_input_ids, q_mask=q_mask, a_mask=a_mask,
                    all_mask=all_mask, targets=None,
                )
                logits_ref = ref_out[0] if isinstance(ref_out, tuple) else ref_out

            # KL preservation loss.
            non_pad = (input_ids != config.pad_token_id)
            kl = kl_preservation_loss(
                logits_ft, logits_ref, non_pad, config.kl_temperature,
            )

            total_loss = ce_loss + config.kl_weight * kl

            # V77 Stage 3: off-distribution conservation KL on probe batches.
            # Probes only contribute input_ids + non-pad mask; NO CE, and the
            # probe labels never enter the loss. Target = frozen backbone
            # distribution over ALL non-pad positions of the full sequence.
            probe_kl = None
            probe_margin = None
            if probe_iter is not None:
                probe_batch = next(probe_iter)
                p_ids, _, p_q_mask, p_a_mask, p_all_mask = _prepare_batch(
                    probe_batch, device,
                )
                logits_probe_ft, _, _ = model(
                    input_ids=p_ids, q_mask=p_q_mask, a_mask=p_a_mask,
                    all_mask=p_all_mask, targets=None,
                )
                with torch.no_grad():
                    ref_probe_out = ref_model(
                        input_ids=p_ids, q_mask=p_q_mask, a_mask=p_a_mask,
                        all_mask=p_all_mask, targets=None,
                    )
                    logits_probe_ref = (
                        ref_probe_out[0]
                        if isinstance(ref_probe_out, tuple) else ref_probe_out
                    )
                probe_non_pad = (p_ids != config.pad_token_id)
                probe_kl = kl_preservation_loss(
                    logits_probe_ft, logits_probe_ref, probe_non_pad,
                    config.probe_kl_temperature,
                )
                total_loss = total_loss + config.probe_kl_weight * probe_kl

            # V77 Stage 5 ("M"): entropy-floor margin at probe answer-start
            # positions. The ft model's answer-start distribution on cue-ended
            # probes must stay AT LEAST as uncertain as the backbone's.
            if config.probe_margin_weight > 0 and probe_iter is not None:
                margin_mask = (
                    probe_batch.get("margin_mask")
                    if isinstance(probe_batch, dict) else None
                )
                if margin_mask is not None:
                    probe_margin = probe_margin_loss(
                        logits_probe_ft, margin_mask.to(device),
                        config.probe_margin_entropy,
                    )
                    total_loss = (total_loss
                                  + config.probe_margin_weight * probe_margin)
                elif not getattr(run_fmsp_finetune, "_margin_warned", False):
                    print(f"  [{model_name}] WARNING: probe_margin_weight > 0 "
                          f"but probe batch has no 'margin_mask'; margin "
                          f"contribution is zero for such batches.")
                    run_fmsp_finetune._margin_warned = True

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Restore frozen scalars bit-exactly (blocks AdamW decoupled wd).
            if frozen_snapshot:
                with torch.no_grad():
                    for name, (mask, values) in frozen_snapshot.items():
                        named_params[name].data[mask] = values

            running_ce += ce_loss.item()
            running_kl += kl.item()
            if probe_kl is not None:
                running_pkl += probe_kl.item()
            if probe_margin is not None:
                running_mgl += probe_margin.item()
            token_acc = aux.get("token_acc", torch.tensor(0.0))
            running_acc += token_acc.item() if isinstance(token_acc, torch.Tensor) else token_acc
            num_batches += 1
            global_step += 1

            if verbose and global_step % config.log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                ans_acc_str = f" ans_acc={ans_acc.item():.3f}" if ans_acc is not None else ""
                pkl_str = f" pkl={probe_kl.item():.4f}" if probe_kl is not None else ""
                mgl_str = f" mgl={probe_margin.item():.4f}" if probe_margin is not None else ""
                print(f"  epoch {epoch} step {global_step}: "
                      f"ce={ce_loss.item():.4f} kl={kl.item():.4f}"
                      f"{pkl_str}{mgl_str} "
                      f"acc={token_acc.item() if isinstance(token_acc, torch.Tensor) else token_acc:.3f}"
                      f"{ans_acc_str} lr={lr_now:.2e}")

        epoch_duration = time.time() - epoch_start
        avg_ce = running_ce / max(num_batches, 1)
        avg_kl = running_kl / max(num_batches, 1)
        avg_pkl = running_pkl / max(num_batches, 1)
        avg_mgl = running_mgl / max(num_batches, 1)
        avg_acc = running_acc / max(num_batches, 1)
        train_loss_history.append({"epoch": epoch, "ce": avg_ce, "kl": avg_kl,
                                   "pkl": avg_pkl, "mgl": avg_mgl, "acc": avg_acc})

        # Validation.
        val_metrics = _fmsp_validate(
            model, val_dataloader, config.pad_token_id, device,
            answer_only_ce=config.answer_only_ce,
        )
        val_loss_history.append(val_metrics)

        if verbose:
            ss_str = f" ss={ss_prob:.3f}" if config.ss_max_prob > 0 else ""
            pkl_str = f" train_pkl={avg_pkl:.4f}" if probe_iter is not None else ""
            mgl_str = (f" train_mgl={avg_mgl:.4f}"
                       if config.probe_margin_weight > 0 else "")
            print(f"[{model_name}] epoch {epoch}: "
                  f"train_ce={avg_ce:.4f} train_kl={avg_kl:.4f}"
                  f"{pkl_str}{mgl_str} "
                  f"train_acc={avg_acc:.3f} | "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"val_acc={val_metrics['val_acc']:.3f} "
                  f"({epoch_duration:.1f}s){ss_str}")

        # Save checkpoint.
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": {
                "train_ce": avg_ce, "train_kl": avg_kl, "train_acc": avg_acc,
                **val_metrics,
            },
            "fmsp_config": config,
            "freeze_summary": freeze_summary,
            "adapter_param_count": adapter_param_count,
        }
        ckpt_path = os.path.join(checkpoint_dir, f"fmsp_epoch_{epoch}.pt")
        torch.save(ckpt, ckpt_path)

    final_val = val_loss_history[-1]["val_loss"] if val_loss_history else float("inf")
    if verbose:
        print(f"\n[{model_name}] FMSP complete. Final val loss: {final_val:.4f}")
        print(f"  Checkpoints in: {checkpoint_dir}/")

    return {
        "fisher_summary": freeze_summary,
        "adapter_param_count": adapter_param_count,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "final_val_loss": final_val,
    }


@torch.no_grad()
def _fmsp_validate(model, val_dataloader, pad_token_id, device,
                   answer_only_ce: bool = False):
    """Validation pass: compute average CE loss + token accuracy.

    When ``answer_only_ce`` is True and the batch carries an ``a_mask``,
    the same answer-region-only CE as training is used, so checkpoint
    selection optimises the training objective.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in val_dataloader:
        input_ids, labels, q_mask, a_mask, all_mask = _prepare_batch(batch, device)
        if labels is None:
            labels = input_ids.clone()

        logits, loss, aux = model(
            input_ids=input_ids, q_mask=q_mask, a_mask=a_mask,
            all_mask=all_mask, targets=labels,
        )
        if answer_only_ce and a_mask is not None:
            loss = _answer_masked_ce(logits, labels, a_mask, pad_token_id)
        total_loss += loss.item()
        token_acc = aux.get("token_acc", torch.tensor(0.0))
        total_acc += token_acc.item() if isinstance(token_acc, torch.Tensor) else token_acc
        num_batches += 1

    model.train()
    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_acc": total_acc / max(num_batches, 1),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("FMSP self-test")
    print("=" * 72)

    # Tiny synthetic model for testing.
    class TinyModel(nn.Module):
        def __init__(self, vocab=100, d=32, seq_len=16):
            super().__init__()
            self.config = type("Cfg", (), {"d_model": d, "pad_token_id": 0})()
            self.embed = nn.Embedding(vocab, d)
            self.out_norm = nn.LayerNorm(d)
            self.linear = nn.Linear(d, d)
            self.vocab = vocab

        def forward(self, input_ids, q_mask=None, a_mask=None,
                    all_mask=None, targets=None):
            h = self.embed(input_ids)
            h = self.linear(h)
            h = self.out_norm(h)
            logits = h @ self.embed.weight.T
            if targets is None:
                return logits, None, {}
            non_pad = (input_ids != 0).float()
            ce = F.cross_entropy(
                logits.reshape(-1, self.vocab),
                targets.reshape(-1),
                ignore_index=0,
                reduction="none",
            ).reshape(input_ids.shape)
            loss = (ce * non_pad).sum() / non_pad.sum().clamp(min=1)
            with torch.no_grad():
                acc = ((logits.argmax(-1) == targets) & non_pad.bool()).float().sum() / non_pad.sum().clamp(min=1)
            return logits, loss, {"token_acc": acc, "ce_loss": loss.detach()}

    torch.manual_seed(42)
    model = TinyModel()
    d = 32

    # 1. Fisher profiling.
    print("\n[1] Fisher profiling:")
    batches = []
    for _ in range(20):
        ids = torch.randint(1, 100, (4, 16))
        batches.append({"input_ids": ids, "labels": ids, "q_mask": None, "a_mask": None})
    fisher = compute_fisher_information(model, batches, num_samples=20, device="cpu")
    print(f"  Computed Fisher for {len(fisher)} params")
    for name, f in fisher.items():
        print(f"  {name}: mean={f.mean():.4e} max={f.max():.4e}")

    # 2. Freeze mask.
    print("\n[2] Freeze mask (freeze_fraction=0.75):")
    mask = build_freeze_mask(fisher, 0.75)
    for name, m in mask.items():
        print(f"  {name}: {int(m.sum())}/{m.numel()} frozen")

    summary = apply_freeze_mask(model, mask)
    print(f"  Summary: {summary}")

    # 3. Adapter.
    print("\n[3] Adapter (rank=4):")
    adapter, handle = attach_adapter(model, d, rank=4)
    print(f"  Adapter params: {sum(p.numel() for p in adapter.parameters())}")

    # Verify adapter is zero-output at init (model unchanged).
    model.eval()
    ids = torch.randint(1, 100, (2, 16))
    with torch.no_grad():
        logits_with_adapter = model(ids, targets=None)[0] if isinstance(model(ids, targets=None), tuple) else model(ids, targets=None)
    handle.remove()
    del model.fmsp_adapter
    with torch.no_grad():
        logits_no_adapter = model(ids, targets=None)[0] if isinstance(model(ids, targets=None), tuple) else model(ids, targets=None)
    max_diff = (logits_with_adapter - logits_no_adapter).abs().max().item()
    print(f"  Max diff (adapter vs no adapter at init): {max_diff:.2e} (should be ~0)")
    assert max_diff < 1e-5, "Adapter not zero at init!"
    print("  [OK] adapter is identity at init")

    # 4. KL loss.
    print("\n[4] KL preservation loss:")
    logits_a = torch.randn(2, 16, 100)
    logits_b = torch.randn(2, 16, 100)
    non_pad = torch.ones(2, 16, dtype=torch.bool)
    kl = kl_preservation_loss(logits_a, logits_b, non_pad)
    print(f"  KL(random, random) = {kl.item():.4f}")
    kl_self = kl_preservation_loss(logits_a, logits_a, non_pad)
    print(f"  KL(x, x) = {kl_self.item():.6f} (should be ~0)")
    assert kl_self.item() < 1e-5, "KL(x,x) != 0"
    print("  [OK]")

    # 5. Gradient masking works.
    print("\n[5] Gradient masking (frozen params get zero grad):")
    # Re-apply mask.
    model_copy = TinyModel()
    mask2 = build_freeze_mask(fisher, 0.5)
    apply_freeze_mask(model_copy, mask2)
    out = model_copy(ids, targets=ids)[1]
    out.backward()
    for name, p in model_copy.named_parameters():
        if p.grad is not None:
            grad_flat = p.grad.flatten()
            mask_flat = mask2[name].flatten().to(p.grad.device)
            frozen_grads = grad_flat[mask_flat]
            if frozen_grads.numel() > 0:
                max_frozen_grad = frozen_grads.abs().max().item()
                assert max_frozen_grad < 1e-10, \
                    f"Frozen param {name} got non-zero grad: {max_frozen_grad}"
    print("  All frozen scalars have zero gradient. [OK]")

    # 6. True freeze (restore-after-step) vs AdamW decoupled weight decay.
    print("\n[6] True freeze: frozen scalars bit-exact after AdamW step:")
    torch.manual_seed(123)
    model_tf = TinyModel()
    pre_step = {n: p.data.clone() for n, p in model_tf.named_parameters()}
    mask_tf = build_freeze_mask(
        {n: torch.rand_like(p) for n, p in model_tf.named_parameters()}, 0.5,
    )
    apply_freeze_mask(model_tf, mask_tf)
    frozen_snapshot = {}
    with torch.no_grad():
        for name, p in model_tf.named_parameters():
            m = mask_tf[name].to(p.device)
            if m.any():
                frozen_snapshot[name] = (m, p.data[m].clone())
    opt = torch.optim.AdamW(model_tf.parameters(), lr=1e-3, weight_decay=0.01)
    ids_tf = torch.randint(1, 100, (4, 16))
    loss_tf = model_tf(ids_tf, targets=ids_tf)[1]
    opt.zero_grad(set_to_none=True)
    loss_tf.backward()
    opt.step()
    with torch.no_grad():
        for name, (m, values) in frozen_snapshot.items():
            p = dict(model_tf.named_parameters())[name]
            p.data[m] = values
    n_frozen_checked = 0
    for name, (m, values) in frozen_snapshot.items():
        p = dict(model_tf.named_parameters())[name]
        assert torch.equal(p.data[m], values), \
            f"Frozen scalars in {name} changed after AdamW step + restore"
        n_frozen_checked += int(m.sum().item())
    changed = any(
        not torch.equal(p.data, pre_step[n])
        for n, p in model_tf.named_parameters()
        if not mask_tf[n].all()
    )
    assert changed, "No trainable scalar changed — AdamW step had no effect"
    print(f"  {n_frozen_checked} frozen scalars bit-identical; "
          f"trainable scalars updated. [OK] true freeze: frozen params "
          f"bit-exact after AdamW step")

    # 7. True freeze with freeze_fraction=0.0 (empty mask -> no-op).
    print("\n[7] True freeze with freeze_fraction=0.0 (empty mask):")
    empty_mask = build_freeze_mask(
        {n: torch.rand_like(p) for n, p in model_tf.named_parameters()}, 0.0,
    )
    empty_snapshot = {}
    with torch.no_grad():
        for name, p in model_tf.named_parameters():
            m = empty_mask[name].to(p.device)
            if m.any():
                empty_snapshot[name] = (m, p.data[m].clone())
    assert len(empty_snapshot) == 0
    with torch.no_grad():
        for name, (m, values) in empty_snapshot.items():
            dict(model_tf.named_parameters())[name].data[m] = values
    print("  Empty snapshot -> restore is a no-op, no crash. [OK]")

    # 8. Answer-only CE (V77-C1): shift semantics + gradient gating.
    print("\n[8] Answer-only CE (shift semantics + gradient gating):")
    torch.manual_seed(7)
    V8 = 20
    # One sequence, no pad in the visible region; input positions 0..7.
    # labels[t] = input_ids[t+1] (final label is pad and ignored).
    ids8 = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 3]])
    labels8 = torch.tensor([[6, 7, 8, 9, 10, 11, 3, 0]])
    a_mask8 = torch.zeros(1, 8, dtype=torch.bool)
    a_mask8[0, 3:] = True  # answer region = input positions 3..7
    # Hand-computed gating: loss at position t kept iff a_mask[t+1], so the
    # trained positions are t = 2..6 (targets = labels[2..6] = 8,9,10,11,3).
    expected_pos = [2, 3, 4, 5, 6]
    logits8 = torch.randn(1, 8, V8)
    # Model predicts the question-region targets (t=0 -> 6, t=1 -> 7) perfectly.
    logits8[0, 0, 6] = 30.0
    logits8[0, 1, 7] = 30.0
    logits8 = logits8.requires_grad_(True)

    ans_ce = _answer_masked_ce(logits8, labels8, a_mask8, pad_token_id=0)
    ce_per = F.cross_entropy(
        logits8.reshape(-1, V8), labels8.reshape(-1),
        ignore_index=0, reduction="none",
    ).reshape(labels8.shape)
    full_ce = (ce_per * (labels8 != 0)).sum() / (labels8 != 0).sum()
    expected_ce = ce_per[0, expected_pos].mean()
    print(f"  ans_ce={ans_ce.item():.6f} expected(mean t=2..6)={expected_ce.item():.6f} "
          f"full_ce={full_ce.item():.6f}")
    assert torch.allclose(ans_ce, expected_ce, atol=1e-6), \
        "shifted mask did not gate exactly the intended target positions"
    assert ans_ce.item() > full_ce.item(), \
        "answer-only CE should exceed full-seq CE when question targets are perfect"
    ans_ce.backward()
    g = logits8.grad
    assert g[0, 0].abs().max().item() == 0.0 and g[0, 1].abs().max().item() == 0.0, \
        "question-position logits received gradient from answer-only CE"
    assert g[0, 7].abs().max().item() == 0.0, \
        "final position (pad target) received gradient"
    assert all(g[0, t].abs().max().item() > 0 for t in expected_pos), \
        "answer-position logits missing gradient"
    print("  Gated positions exactly t=2..6; question-position grads zero; "
          "ans_ce > full_ce. [OK]")

    # 9. Probe-KL (V77 Stage 3): zero for identical models, grad flows,
    #    weight=0.0 path is byte-identical to the no-probe path.
    print("\n[9] Probe-KL (off-distribution conservation, V77 Stage 3):")
    torch.manual_seed(99)
    m9 = TinyModel()
    probe_ids9 = torch.randint(1, 100, (2, 16))
    probe_non_pad9 = (probe_ids9 != 0)

    # (a) probe_kl == 0 when ft and ref are the same model.
    logits_ft9 = m9(probe_ids9, targets=None)[0]
    with torch.no_grad():
        logits_ref9 = m9(probe_ids9, targets=None)[0]
    pkl_self = kl_preservation_loss(logits_ft9, logits_ref9, probe_non_pad9, 1.0)
    print(f"  (a) probe_kl(ft, ft) = {pkl_self.item():.6f} (should be ~0)")
    assert pkl_self.item() < 1e-6, "probe-KL(x,x) != 0"

    # (b) probe gradient flows: probe-KL-only backward -> nonzero grads on
    #     trainable params (ft perturbed away from a frozen ref copy).
    ref9 = copy.deepcopy(m9)
    for p in ref9.parameters():
        p.requires_grad = False
    with torch.no_grad():
        m9.linear.weight.add_(0.05)  # move ft distribution off the ref
    logits_ft9b = m9(probe_ids9, targets=None)[0]
    with torch.no_grad():
        logits_ref9b = ref9(probe_ids9, targets=None)[0]
    pkl_b = kl_preservation_loss(logits_ft9b, logits_ref9b, probe_non_pad9, 1.0)
    m9.zero_grad(set_to_none=True)
    pkl_b.backward()
    grads9 = [p.grad for p in m9.parameters() if p.grad is not None]
    max_g = max(g.abs().max().item() for g in grads9)
    print(f"  (b) probe_kl = {pkl_b.item():.4f}, max |grad| on ft params = {max_g:.3e}")
    assert pkl_b.item() > 0, "perturbed ft should have positive probe-KL"
    assert max_g > 0, "probe-KL backward produced no gradient on trainable params"

    # (c) probe_kl_weight=0.0: no probe loader needed, no error, and the
    #     result is byte-identical with or without a (dummy) probe loader.
    def _tiny_run9(probe_loader):
        torch.manual_seed(5)
        mdl = TinyModel()
        qa9 = []
        for _ in range(6):
            ids = torch.randint(1, 100, (2, 16))
            am = torch.zeros(2, 16, dtype=torch.bool)
            am[:, 8:] = True
            qa9.append({"input_ids": ids, "labels": torch.cat(
                [ids[:, 1:], torch.zeros(2, 1, dtype=torch.long)], dim=1),
                "q_mask": ~am, "a_mask": am})
        cfg9 = FMSPConfig(
            freeze_fraction=0.5, adapter_rank=2, kl_weight=0.05,
            fisher_num_samples=2, learning_rate=1e-3, max_epochs=1,
            log_interval=999, seed=7, pad_token_id=0,
            probe_kl_weight=0.0,
        )
        run_fmsp_finetune(
            model=mdl, pretrain_dataloader=qa9, finetune_dataloader=qa9,
            val_dataloader=qa9, config=cfg9, device="cpu",
            checkpoint_dir="/tmp/fmsp_selftest9", model_name="selftest9",
            d_model=32, verbose=False, probe_dataloader=probe_loader,
        )
        return {n: p.data.clone() for n, p in mdl.named_parameters()}

    params_none = _tiny_run9(None)
    dummy_probe = [{"input_ids": torch.randint(1, 100, (2, 16)),
                    "labels": torch.zeros(2, 16, dtype=torch.long),
                    "q_mask": torch.ones(2, 16, dtype=torch.bool),
                    "a_mask": torch.ones(2, 16, dtype=torch.bool)}]
    params_dummy = _tiny_run9(dummy_probe)
    assert params_none.keys() == params_dummy.keys()
    for n in params_none:
        assert torch.equal(params_none[n], params_dummy[n]), \
            f"probe_kl_weight=0.0 path diverged with/without probe loader: {n}"
    print("  (c) probe_kl_weight=0.0: no loader required, params byte-identical "
          "with/without a dummy probe loader. [OK]")

    # (d) probe_kl_weight>0 without a probe loader must fail loud.
    try:
        _cfg9d = FMSPConfig(probe_kl_weight=1.0)
        run_fmsp_finetune(
            model=TinyModel(), pretrain_dataloader=[], finetune_dataloader=[],
            val_dataloader=[], config=_cfg9d, device="cpu",
            checkpoint_dir="/tmp/fmsp_selftest9d", model_name="selftest9d",
            d_model=32, verbose=False, probe_dataloader=None,
        )
        raise AssertionError("probe_kl_weight>0 without probe_dataloader did not raise")
    except ValueError:
        print("  (d) probe_kl_weight>0 without probe_dataloader -> ValueError. [OK]")

    # 10. Probe margin loss (V77 Stage 5, "M"): entropy floor at masked
    #     answer-start positions.
    print("\n[10] Probe margin loss (entropy floor, V77 Stage 5):")
    V10 = 100
    h_target10 = 4.0  # ln(100) ~= 4.605, so uniform logits clear the target
    mmask10 = torch.zeros(2, 16, dtype=torch.bool)
    mmask10[0, 7] = True
    mmask10[1, 12] = True

    # (a) L_margin == 0 when masked-position entropy >= H_target (uniform).
    logits_uniform = torch.zeros(2, 16, V10)
    mgl_a = probe_margin_loss(logits_uniform, mmask10, h_target10)
    print(f"  (a) L_margin(uniform, H_target={h_target10}) = {mgl_a.item():.6f} "
          f"(should be 0)")
    assert mgl_a.item() == 0.0, "margin should be 0 when entropy >= H_target"

    # (b) L_margin > 0 and gradient flows when entropy < H_target (peaked).
    logits_peaked = torch.zeros(2, 16, V10, requires_grad=True)
    with torch.no_grad():
        logits_peaked[0, 7, 3] = 20.0   # sharp at a masked position
        logits_peaked[1, 12, 9] = 20.0  # sharp at the other masked position
        logits_peaked[0, 0, 5] = 20.0   # sharp at an UNMASKED position (ignored)
    mgl_b = probe_margin_loss(logits_peaked, mmask10, h_target10)
    mgl_b.backward()
    g10 = logits_peaked.grad
    h_peaked = -(torch.softmax(torch.tensor([20.0] + [0.0] * (V10 - 1)), -1)
                 * torch.log_softmax(torch.tensor([20.0] + [0.0] * (V10 - 1)), -1)).sum()
    expected_b = (h_target10 - h_peaked).clamp(min=0)
    print(f"  (b) L_margin(peaked) = {mgl_b.item():.4f} "
          f"(expected ~= {expected_b.item():.4f}), "
          f"grad@masked={g10[0, 7].abs().max().item():.3e}, "
          f"grad@unmasked={g10[0, 0].abs().max().item():.3e}")
    assert mgl_b.item() > 0, "peaked masked positions should give positive margin"
    assert abs(mgl_b.item() - expected_b.item()) < 1e-3, \
        "margin value mismatch vs hand-computed entropy deficit"
    assert g10[0, 7].abs().max().item() > 0, "no gradient at masked position"
    assert g10[1, 12].abs().max().item() > 0, "no gradient at masked position"
    assert g10[0, 0].abs().max().item() == 0.0, \
        "unmasked position received margin gradient"
    print("  (b) positive margin, value matches hand computation, "
          "gradient flows only at masked positions. [OK]")

    # (c) margin disabled (weight 0.0): byte-identical with/without
    #     margin_mask in the probe batches.
    def _tiny_run10(with_margin_mask):
        torch.manual_seed(11)
        mdl = TinyModel()
        qa10 = []
        for _ in range(6):
            ids = torch.randint(1, 100, (2, 16))
            am = torch.zeros(2, 16, dtype=torch.bool)
            am[:, 8:] = True
            qa10.append({"input_ids": ids, "labels": torch.cat(
                [ids[:, 1:], torch.zeros(2, 1, dtype=torch.long)], dim=1),
                "q_mask": ~am, "a_mask": am})
        probe_ids10 = torch.randint(1, 100, (2, 16))
        probe10 = {"input_ids": probe_ids10,
                   "labels": torch.zeros(2, 16, dtype=torch.long),
                   "q_mask": torch.ones(2, 16, dtype=torch.bool),
                   "a_mask": torch.ones(2, 16, dtype=torch.bool)}
        if with_margin_mask:
            mm = torch.zeros(2, 16, dtype=torch.bool)
            mm[0, 15] = True
            probe10["margin_mask"] = mm
        cfg10 = FMSPConfig(
            freeze_fraction=0.5, adapter_rank=2, kl_weight=0.05,
            fisher_num_samples=2, learning_rate=1e-3, max_epochs=1,
            log_interval=999, seed=7, pad_token_id=0,
            probe_kl_weight=0.5, probe_margin_weight=0.0,
        )
        run_fmsp_finetune(
            model=mdl, pretrain_dataloader=qa10, finetune_dataloader=qa10,
            val_dataloader=qa10, config=cfg10, device="cpu",
            checkpoint_dir="/tmp/fmsp_selftest10", model_name="selftest10",
            d_model=32, verbose=False, probe_dataloader=[probe10],
        )
        return {n: p.data.clone() for n, p in mdl.named_parameters()}

    params_no_mm = _tiny_run10(False)
    params_with_mm = _tiny_run10(True)
    assert params_no_mm.keys() == params_with_mm.keys()
    for n in params_no_mm:
        assert torch.equal(params_no_mm[n], params_with_mm[n]), \
            f"probe_margin_weight=0.0 path diverged with/without margin_mask: {n}"
    print("  (c) probe_margin_weight=0.0: params byte-identical with/without "
          "margin_mask. [OK]")

    # (d) margin enabled but probe batch lacks margin_mask: tolerated, zero
    #     contribution, one-time warning, training completes.
    torch.manual_seed(13)
    mdl10d = TinyModel()
    qa10d = []
    for _ in range(6):
        ids = torch.randint(1, 100, (2, 16))
        am = torch.zeros(2, 16, dtype=torch.bool)
        am[:, 8:] = True
        qa10d.append({"input_ids": ids, "labels": torch.cat(
            [ids[:, 1:], torch.zeros(2, 1, dtype=torch.long)], dim=1),
            "q_mask": ~am, "a_mask": am})
    run_fmsp_finetune._margin_warned = False
    cfg10d = FMSPConfig(
        freeze_fraction=0.5, adapter_rank=2, kl_weight=0.05,
        fisher_num_samples=2, learning_rate=1e-3, max_epochs=1,
        log_interval=999, seed=7, pad_token_id=0,
        probe_kl_weight=0.5, probe_margin_weight=0.5, probe_margin_entropy=4.0,
    )
    metrics10d = run_fmsp_finetune(
        model=mdl10d, pretrain_dataloader=qa10d, finetune_dataloader=qa10d,
        val_dataloader=qa10d, config=cfg10d, device="cpu",
        checkpoint_dir="/tmp/fmsp_selftest10d", model_name="selftest10d",
        d_model=32, verbose=False,
        probe_dataloader=[{"input_ids": torch.randint(1, 100, (2, 16)),
                           "labels": torch.zeros(2, 16, dtype=torch.long),
                           "q_mask": torch.ones(2, 16, dtype=torch.bool),
                           "a_mask": torch.ones(2, 16, dtype=torch.bool)}],
    )
    assert run_fmsp_finetune._margin_warned, "missing-margin_mask warning not raised"
    assert metrics10d["train_loss_history"][0]["mgl"] == 0.0, \
        "margin contribution should be zero when margin_mask is absent"
    print("  (d) margin enabled + missing margin_mask: tolerated, mgl=0, "
          "warning fired. [OK]")

    # (e) margin enabled with margin_mask: mgl logged and > 0 when the model
    #     is sharper than H_target at masked positions.
    torch.manual_seed(17)
    mdl10e = TinyModel()
    qa10e = [dict(b) for b in qa10d]
    mm10e = torch.ones(2, 16, dtype=torch.bool)  # mask everywhere -> some pos is sharp
    metrics10e = run_fmsp_finetune(
        model=mdl10e, pretrain_dataloader=qa10e, finetune_dataloader=qa10e,
        val_dataloader=qa10e, config=FMSPConfig(
            freeze_fraction=0.5, adapter_rank=2, kl_weight=0.05,
            fisher_num_samples=2, learning_rate=1e-3, max_epochs=1,
            log_interval=999, seed=7, pad_token_id=0,
            probe_kl_weight=0.5, probe_margin_weight=0.5,
            probe_margin_entropy=5.0,  # above a random net's entropy -> deficit
        ), device="cpu",
        checkpoint_dir="/tmp/fmsp_selftest10e", model_name="selftest10e",
        d_model=32, verbose=False,
        probe_dataloader=[{"input_ids": torch.randint(1, 100, (2, 16)),
                           "labels": torch.zeros(2, 16, dtype=torch.long),
                           "q_mask": torch.ones(2, 16, dtype=torch.bool),
                           "a_mask": torch.ones(2, 16, dtype=torch.bool),
                           "margin_mask": mm10e}],
    )
    assert metrics10e["train_loss_history"][0]["mgl"] > 0.0, \
        "margin should be positive when H_target exceeds model entropy"
    print("  (e) margin enabled with margin_mask: mgl > 0 when model is "
          "sharper than H_target. [OK]")

    print("\n" + "=" * 72)
    print("ALL FMSP SELF-TESTS PASSED")
    print("=" * 72)
