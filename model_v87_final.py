"""MicroMixer-4 V87 Final -- the project's final clean architecture.

V86's final census (V86_README.md S5) crowned V83-RPG (the "RPG" arm of the
V83 CCD-Mixer) the overall champion: chatter d2 0.883 (family record),
full-988 EM 563.7, qrel hijack 4.0%, OOD hijack 6.8% -- the only arm holding
all four guards simultaneously. V87 Final = V83-RPG as the SINGLE
architecture (no arm variants), context window 1024, scaled to 6 parameter
budgets.

== What V87 Final IS ==

V83-RPG verbatim, at 6 scales. The mechanism is unchanged from V83 S2:

    u    = Linear(d -> 2d)(x)
    v, g = u.chunk(2, dim=-1)                    # each d-dim
    v    = rope(v)                               # V76 full-RoPE on v
    g    = rope(g)                               # V76 full-RoPE on g
    for d in (1, 2, 4, 8):
        y_d = F.conv1d(pad_(k-1)*d(v), W_conv, b_conv, dilation=d)  # SAME weights
    w(t) = softmax(Linear_dil(x)_t / tau)        # per-position 4-way gate
    v    = sum_d w_d(t) * y_d(t)                  # time-varying effective filter
    out  = W_o(v * g)                             # W_o zero-init (silent at init)

Content-gated mixture of shared-weight dilated causal convolutions: the
token-mix is non-LTI (time-varying), which is the mechanism V80's LTI
analysis identified as necessary to break periodic orbits without RoPE's
byte-identity scattering. The four branches share one depthwise kernel
(0 extra conv params); only the gate (Linear d->4) and a learnable
temperature tau are new (4d + 5 params/block over the V76 base).

== What V87 Final REMOVES vs V83 ==

Everything except RPG:
  - NO arm parameter (RPG is the only RoPE pattern, hardcoded).
  - NO NcNg arm (V82-NcNg minimal-rotation pattern).
  - NO NcNgA arm (Adaptive Orthogonality Angle block).
  - NO AOARTMBlock class (parallel angle-blended mixes).
  - NO arm validation in the model or token-mix constructor.

The chassis is otherwise bit-identical to V83-RPG: V72-GLU backbone +
V71 RTMBlock stack + SwiGLU h=swiglu_hidden(d) + ReMixerLayer sidecar +
GLCTokenMixCCD(arm="RPG") token-mix swap + RMSNorm pre-norm + tied byte
head + W_o zero-init + dil_gate zero-init + log_tau=0.

== Perfect ablation property (inherited from V83) ==

With ``force_dilation = 0`` the gate becomes one-hot on the dil=1 branch
and V87 is BIT-IDENTICAL to its V76-RPG parent with matched weights.
Self-test [CCD] hard-asserts diff == 0.0 in this mode and diff > 1e-3
with the gate free (mechanism active).

== Preset table ==

All presets use max_seq_len=1024, label_dim=16, pool_heads=4, pool_rank=0
(auto = max(4, d//8)), expansion_channel=2, the V83 CCD dilations (1,2,4,8),
and arm="RPG" (hardcoded). d_model is a multiple of 16 for 1M/500K/300K/
100K and a multiple of 8 for 50K/10K (both divisible by 4, satisfying
GLCTokenMixGCR's constraint).

  | size  | d_model | num_blocks | glc_kernel | label_dim | pool_heads | pool_rank | params    |
  |-------|---------|------------|------------|-----------|------------|-----------|-----------|
  | 1M    |     128 |          7 |        129 |        16 |          4 |   auto(16)|   996,873 |
  | 500K  |      96 |          6 |         97 |        16 |          4 |   auto(12)|   491,742 |
  | 300K  |      80 |          5 |         81 |        16 |          4 |   auto(10)|   292,525 |
  | 100K  |      48 |          4 |         65 |        16 |          4 |    auto(6)|    95,084 |
  | 50K   |      32 |          4 |         65 |        16 |          4 |    auto(4)|    48,684 |
  | 10K   |      16 |          2 |         33 |        16 |          4 |    auto(4)|     9,666 |

Param formula (exact for all presets):

    swiglu_hidden(d)   = (4*d*d + 2*d) // (3*d + 2)
    token_mix/block    = 3*d*d + d*k + 8*d + 5          (CCD: base + 4d + 5)
    channel_mix/block  = 3*d*h + 2*h + d                 (SwiGLU)
    remixer/block      = 3*d + 2 + (K+r)*(d+1) + r*d     (norm + kappa + saliency
                                                          + pool_proj + pool_down
                                                          + pool_up)
    per_block          = token_mix + 2*d (norms) + channel_mix + remixer
    total              = 257*d + num_blocks * per_block  (embed 256*d + out_norm d)

The 1M preset reproduces V83-RPG EXACTLY (996,873 params, d=128, N=7, k=129,
all chassis defaults). Smaller presets shrink d_model (multiples of 16 down
to 100K, 8 below), num_blocks, and glc_kernel (odd; k=129 is disproportionate
for tiny models). Each preset is UNDER its budget and within ~5% of it
(except 10K where the tied embedding alone is 256*d = 4096 params -- the
structural floor; 9,666 is as close under 10,000 as the architecture allows).

10K design compromise: num_blocks=2 is the minimum depth that preserves the
CCD token-mix + SwiGLU + ReMixerLayer block structure. Fewer blocks would
drop below the ReMixerLayer sidecar's minimum useful depth; the 2-block stack
gives a per-block receptive field of (k-1)*8 = 256 at dil=8, covering 256 of
1024 context positions per block. The kernel k=33 (not 17) was chosen to land
at 9,666 params (96.7% of budget) rather than 9,154 (91.5%) -- the wider
receptive field is worth the 513-param cost.

== Init behavior (inherited) ==

W_o zero-init (inherited) makes the token-mix silent at init, so V87
forward is bit-identical to its token-mix-zeroed self (T1). The CCD gate is
zero-init -> uniform 1/4 mixture at step 0 (neutral over timescales; the
"no human prior" stance). log_tau = 0 -> tau = 1. kappa = 0 and pool_up = 0
-> ReMixerLayer sidecar silent at init.

== Pure MLP-Mixer ==

No attention, no recurrence, no SSM. Depthwise causal convs (left-pad only),
SwiGLU channel mixing, RMSNorm pre-norm, tied byte head. Causality is exact:
every branch left-pads (k-1)*d and the gate is pointwise in position.

Self-test: per-preset T1/T2/Causality (6 presets) + cross-preset Interface /
Module-scan / Generation / CCD-equivalence, runnable via
``python src/model_v87_final.py``.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from dataclasses import dataclass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_v37_qag import RMSNorm
from src.model_v71_rtm import V71Config, count_parameters
from src.model_v72_glumixer import SwiGLUChannelMLP
from src.model_v67_contentpool import ReMixerLayer
from src.model_v76_rpg import (
    MicroMixerV76RPG,
    v76_rpg_1m,
)
from src.model_v83_ccd import GLCTokenMixCCD


# ============================================================================
# Config
# ============================================================================

_V87_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

_V87_PRESETS: tuple[str, ...] = ("1M", "500K", "300K", "100K", "50K", "10K")

# Exact param counts (hard-asserted in self-test; computed once and frozen).
#   1M:   996,873  (V83-RPG exact reproduction: 993,254 chassis + 3,619 CCD gates)
#   500K: 491,742
#   300K: 292,525
#   100K:  95,084
#   50K:   48,684
#   10K:    9,666
_V87_EXPECTED_PARAMS: dict[str, int] = {
    "1M":   996_873,
    "500K": 491_742,
    "300K": 292_525,
    "100K":  95_084,
    "50K":   48_684,
    "10K":    9_666,
}

# Budget ceilings (each preset's total must be strictly UNDER its budget).
_V87_BUDGETS: dict[str, int] = {
    "1M":   1_000_000,
    "500K":   500_000,
    "300K":   300_000,
    "100K":   100_000,
    "50K":     50_000,
    "10K":     10_000,
}

# Per-preset (d_model, num_blocks, glc_kernel) -- the ONLY knobs that vary.
_V87_PRESET_SPECS: dict[str, tuple[int, int, int]] = {
    "1M":   (128, 7, 129),
    "500K": ( 96, 6,  97),
    "300K": ( 80, 5,  81),
    "100K": ( 48, 4,  65),
    "50K":  ( 32, 4,  65),
    "10K":  ( 16, 2,  33),
}

# Per-block token-mix params formula: 3*d*d + d*k + 8*d + 5 (CCD: base + gate).
# state_dict key count: 3 (embed + byte_labels + out_norm) + 27 * num_blocks.
#   Per-block keys: norm1(1) + token_mix(7: proj_in.w/b, conv.w/b, W_o.w/b,
#                   dil_gate.w/b, log_tau) + norm2(1) + channel_mix(6) +
#                   remixer(11) = 27.


@dataclass
class V87Config(V71Config):
    """Configuration for MicroMixerV87Final.

    Subclasses V71Config (for interface parity with the V71->V83 family) and
    adds ``ccd_dilations`` to select the candidate dilation set for the
    content-controlled mixture. Default (1, 2, 4, 8); dil=1 MUST be present
    for the [CCD] bit-equivalence ablation to be meaningful.

    NO ``arm`` parameter (V83 had one): RPG is the only RoPE pattern,
    hardcoded in MicroMixerV87Final.__init__. The V82/V83 NcNg and NcNgA
    code paths are gone.
    """
    ccd_dilations: tuple = (1, 2, 4, 8)


# ============================================================================
# V87 Final model
# ============================================================================

class MicroMixerV87Final(MicroMixerV76RPG):
    """V87 Final: V83-RPG as the SINGLE architecture, scaled to 6 budgets.

    Construction mirrors V83 exactly minus the NcNg/NcNgA code paths:
    MicroMixerV76RPG.__init__ is called DIRECTLY (bypassing V82/V83 arm
    validation), building the full V76 chassis (V72-GLU backbone + V71
    RTMBlock stack + SwiGLUChannelMLP h=swiglu_hidden(d) + ReMixerLayer
    sidecar + GLCTokenMixRPG token-mix swap). The RPG token-mixes are then
    discarded and replaced by GLCTokenMixCCD with arm="RPG" (hardcoded --
    the V76 full-RoPE pattern on v AND g).

    Interface contract (identical to V71->V83):
      - forward(input_ids, q_mask, a_mask, all_mask, targets)
        -> (logits, loss|None, aux); masks accepted and IGNORED.
      - generate(input_ids, q_mask, a_mask, all_mask, *, max_new_tokens,
        temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size,
        eos_token_id) -> full sequence.
      - model.config, model.out_norm, model.embed are public attributes.
      - Tied output head: logits = out_norm(h) @ embed.weight.T.
      - Model is copy.deepcopy-able and state_dict strict-loadable.
      - V76-RPG checkpoints strict=False-load into V87-1M (missing keys are
        exactly the 21 CCD gate keys: 7 blocks x [dil_gate.weight,
        dil_gate.bias, log_tau]).
    """

    def __init__(self, config: V87Config):
        if tuple(config.ccd_dilations)[0] != 1:
            raise ValueError(
                f"ccd_dilations must start at 1 (ablation anchor); got "
                f"{config.ccd_dilations}"
            )
        if len(tuple(config.ccd_dilations)) < 2:
            raise ValueError(
                f"ccd_dilations must have >= 2 candidates; got "
                f"{config.ccd_dilations}"
            )
        # Build the V76 chassis directly (V82's __init__ rejects non-V82 arms;
        # V83's __init__ adds arm validation we do not need). This constructs
        # the RPG token-mixes, which we discard below.
        MicroMixerV76RPG.__init__(self, config)

        # Swap token_mix to CCD with arm="RPG" (hardcoded -- no arm parameter).
        # GLCTokenMixCCD.__init__ chain: -> GLCTokenMixGCR -> GLCTokenMixRPG ->
        # GLCTokenMixRPC -> GLCTokenMix. Builds proj_in/conv/W_o (house init,
        # W_o zeroed), 64-pair gate RoPE tables (for v and g), 32-pair state
        # tables (unused by RPG but allocated by the GCR parent), and the CCD
        # dil_gate (zero-init) + log_tau (=0).
        for b in self.blocks:
            b.token_mix = GLCTokenMixCCD(
                config.d_model, config.glc_kernel, config.max_seq_len,
                "RPG", tuple(config.ccd_dilations),
            )


# ============================================================================
# Presets
# ============================================================================

def v87_final_1m() -> V87Config:
    """996,873 params. EXACT V83-RPG reproduction (d=128, N=7, k=129).

    The champion preset: chatter d2 0.883, full-988 EM 563.7, qrel hijack
    4.0%, OOD hijack 6.8% under the V76 recipe (3-seed).
    """
    return V87Config(
        tmix="glc", d_model=128, num_blocks=7, glc_kernel=129, max_seq_len=1024,
    )


def v87_final_500k() -> V87Config:
    """491,742 params (d=96, N=6, k=97). 98.3% of the 500K budget."""
    return V87Config(
        tmix="glc", d_model=96, num_blocks=6, glc_kernel=97, max_seq_len=1024,
    )


def v87_final_300k() -> V87Config:
    """292,525 params (d=80, N=5, k=81). 97.5% of the 300K budget."""
    return V87Config(
        tmix="glc", d_model=80, num_blocks=5, glc_kernel=81, max_seq_len=1024,
    )


def v87_final_100k() -> V87Config:
    """95,084 params (d=48, N=4, k=65). 95.1% of the 100K budget."""
    return V87Config(
        tmix="glc", d_model=48, num_blocks=4, glc_kernel=65, max_seq_len=1024,
    )


def v87_final_50k() -> V87Config:
    """48,684 params (d=32, N=4, k=65). 97.4% of the 50K budget."""
    return V87Config(
        tmix="glc", d_model=32, num_blocks=4, glc_kernel=65, max_seq_len=1024,
    )


def v87_final_10k() -> V87Config:
    """9,666 params (d=16, N=2, k=33). 96.7% of the 10K budget.

    Structural floor: the tied embedding alone is 256*16 = 4,096 params
    (42.4% of the total). 2 blocks is the minimum depth preserving the
    CCD + SwiGLU + ReMixerLayer block structure. See module docstring for
    the full design-compromise discussion.
    """
    return V87Config(
        tmix="glc", d_model=16, num_blocks=2, glc_kernel=33, max_seq_len=1024,
    )


# ============================================================================
# Self-test helpers
# ============================================================================

def _unzero_w_o(model: MicroMixerV87Final, seed: int = 7) -> None:
    """Un-zero all zero-init W_o at a fixed seed (exercises the CCD paths).

    At init W_o=0 makes the entire token_mix output 0, killing the gradient
    to proj_in / conv / dil_gate. Un-zeroing exercises the CCD + RoPE paths.
    """
    torch.manual_seed(seed)
    for block in model.blocks:
        tm = block.token_mix
        nn.init.xavier_uniform_(tm.W_o.weight)
        nn.init.zeros_(tm.W_o.bias)


def _monkeypatch_token_mix_zero(model: MicroMixerV87Final) -> list:
    """Monkeypatch each block's GLCTokenMixCCD forward to return zeros.

    Returns list of (block_idx, original_forward) for restoration.
    """
    patched = []
    for i, block in enumerate(model.blocks):
        tm = block.token_mix
        orig = tm.forward

        def _zero(self, x):
            return torch.zeros_like(x)
        tm.forward = types.MethodType(_zero, tm)
        patched.append((i, orig))
    return patched


def _restore_token_mix(model: MicroMixerV87Final, patched: list) -> None:
    for i, orig in patched:
        model.blocks[i].token_mix.forward = orig


def _set_force_dilation(model: MicroMixerV87Final, idx: int | None) -> None:
    """Set the CCD ablation hook on every block's token_mix.

    idx=0 -> one-hot gate on dil=1 branch (bit-identical to V76 parent).
    idx=None -> free gate (CCD mechanism active).
    """
    for b in model.blocks:
        b.token_mix.force_dilation = idx


def _set_force_uniform(model: MicroMixerV87Final, flag: bool = True) -> None:
    """Set the uniform-1/4 control hook on every block's token_mix.

    flag=True -> gate frozen at the uniform mixture ones(K)/K at every
    position (LTI, receptive-field preserved, no content-dependent selection).
    flag=False -> free gate (CCD mechanism active; normal V87 behavior).
    """
    for b in model.blocks:
        b.token_mix.force_uniform = flag


def _expected_tmix_params(d: int, k: int) -> int:
    """Per-block CCD token-mix param count (formula, exact for all presets).

    proj_in (2d*d + 2d) + conv (d*k + d) + W_o (d*d + d) + dil_gate (4d + 4)
    + log_tau (1) = 3d^2 + dk + 8d + 5.
    """
    return 3 * d * d + d * k + 8 * d + 5


def _expected_sd_keys(num_blocks: int) -> int:
    """state_dict key count: 3 model-level + 27 per block."""
    return 3 + 27 * num_blocks


# ============================================================================
# Self-test
# ============================================================================

def _run_preset_tests(size: str, V: int) -> int:
    """Run T1/T2/Causality for a single V87 preset. Returns pass count."""
    passes = 0
    cfg = _PRESET_FNS[size]()
    d, N, k = _V87_PRESET_SPECS[size]
    exp_params = _V87_EXPECTED_PARAMS[size]
    budget = _V87_BUDGETS[size]
    exp_tmix = _expected_tmix_params(d, k)

    print(f"\n{'='*72}")
    print(f"  PRESET: {size}  (d={d}, blocks={N}, glc_kernel={k}, "
          f"dilations={_V87_DILATIONS}, SwiGLU h=swiglu_hidden({d}))")
    print(f"{'='*72}")

    # === T1: Param count + budget + zero-init + bit-identical init + finite grads ===
    print(f"\n[T1-{size}] Param count ({exp_params:,}) + budget (<{budget:,}) + "
          f"zero-init + bit-identical init + finite grads:")

    m = MicroMixerV87Final(cfg)
    n = count_parameters(m)
    ok_params = (n == exp_params)
    ok_budget = (n < budget)
    print(f"  params: {n:>10,}  (expected {exp_params:>10,})  "
          f"[{'OK' if ok_params else 'FAIL'}]  "
          f"budget: [{'OK' if ok_budget else 'OVER'}]  "
          f"({n/budget*100:.1f}% of {budget:,})")
    assert ok_params, f"[{size}] param count mismatch: {n} != {exp_params}"
    assert ok_budget, f"[{size}] over budget: {n} >= {budget}"

    # Per-block token-mix accounting (formula-derived, exact for all presets).
    tm0 = m.blocks[0].token_mix
    tm_params = sum(p.numel() for p in tm0.parameters())
    print(f"  token-mix/block: {tm_params:,}  (expected {exp_tmix:,})  "
          f"[{'OK' if tm_params == exp_tmix else 'FAIL'}]")
    assert tm_params == exp_tmix, (
        f"[{size}] token-mix params {tm_params} != {exp_tmix}"
    )

    # Zero-init conditions: W_o=0, kappa=0, pool_up=0, dil_gate=0, log_tau=0.
    all_wo_zero = all(
        b.token_mix.W_o.weight.abs().max().item() == 0.0
        and b.token_mix.W_o.bias.abs().max().item() == 0.0
        for b in m.blocks
    )
    all_kappa_zero = all(
        b.remixer.kappa.abs().item() == 0.0 for b in m.blocks
    )
    all_pool_up_zero = all(
        b.remixer.pool_up.weight.abs().max().item() == 0.0
        and b.remixer.pool_up.bias.abs().max().item() == 0.0
        for b in m.blocks
    )
    all_gate_zero = all(
        b.token_mix.dil_gate.weight.abs().max().item() == 0.0
        and b.token_mix.dil_gate.bias.abs().max().item() == 0.0
        and b.token_mix.log_tau.abs().max().item() == 0.0
        for b in m.blocks
    )
    print(f"  W_o=0: {all_wo_zero}; kappa=0: {all_kappa_zero}; "
          f"pool_up=0: {all_pool_up_zero}; dil_gate=0 & log_tau=0: {all_gate_zero}")
    assert all_wo_zero, f"[{size}] W_o not zero at init"
    assert all_kappa_zero, f"[{size}] kappa not zero at init"
    assert all_pool_up_zero, f"[{size}] pool_up not zero at init"
    assert all_gate_zero, f"[{size}] CCD gate not zero-init"

    # Bit-identical: forward at init == forward with token-mix zeroed.
    m.eval()
    test_ids = torch.randint(4, V, (2, 64))
    with torch.no_grad():
        logits_normal = m(test_ids, None, None, None, None)[0]
    patched = _monkeypatch_token_mix_zero(m)
    with torch.no_grad():
        logits_zeroed = m(test_ids, None, None, None, None)[0]
    _restore_token_mix(m, patched)
    init_diff = (logits_normal - logits_zeroed).abs().max().item()
    print(f"  bit-identical init (token-mix zeroed): max|diff| = {init_diff:.2e}")
    assert init_diff == 0.0, f"[{size}] not bit-identical at init: {init_diff}"

    # Finite grads.
    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat(
        [ids[:, 1:], torch.full((B, 1), 0, dtype=torch.long)], dim=1,
    )
    m.train()
    m.zero_grad()
    logits, loss, aux = m(ids, None, None, None, targets)
    assert logits.shape == (B, S, V)
    assert torch.isfinite(loss).item()
    loss.backward()
    finite = all(
        torch.isfinite(p.grad).all().item()
        for p in m.parameters() if p.grad is not None
    )
    assert finite, f"[{size}] non-finite gradients"
    print(f"  loss={loss.item():.4f} acc={aux['token_acc'].item():.4f} "
          f"finite_grads={finite}")
    print(f"  [T1 PASS]")
    passes += 1

    # === T2: Grad-flow (conv / proj_in / W_o / dil_gate, W_o un-zeroed) ===
    print(f"\n[T2-{size}] Grad-flow (conv / proj_in / W_o / dil_gate, "
          f"W_o un-zeroed):")
    m2 = MicroMixerV87Final(cfg)
    _unzero_w_o(m2, seed=7)
    m2.train()
    m2.zero_grad()
    m2(ids, None, None, None, targets)[1].backward()
    finite2 = all(
        torch.isfinite(p.grad).all().item()
        for p in m2.parameters() if p.grad is not None
    )
    assert finite2, f"[{size}] non-finite grads after W_o un-zero"

    conv_grads = [
        b.token_mix.conv.conv.weight.grad.abs().max().item() for b in m2.blocks
    ]
    proj_in_grads = [
        b.token_mix.proj_in.weight.grad.abs().max().item() for b in m2.blocks
    ]
    wo_grads = [
        b.token_mix.W_o.weight.grad.abs().max().item() for b in m2.blocks
    ]
    gate_grads = [
        b.token_mix.dil_gate.weight.grad.abs().max().item() for b in m2.blocks
    ]
    max_conv = max(conv_grads)
    max_proj = max(proj_in_grads)
    max_wo = max(wo_grads)
    max_gate = max(gate_grads)
    print(f"  conv grads:     {[f'{g:.2e}' for g in conv_grads]}")
    print(f"  proj_in grads:  {[f'{g:.2e}' for g in proj_in_grads]}")
    print(f"  W_o grads:      {[f'{g:.2e}' for g in wo_grads]}")
    print(f"  dil_gate grads: {[f'{g:.2e}' for g in gate_grads]}")
    assert max_conv > 0, f"[{size}] conv grad zero (CCD conv path not wired)"
    assert max_proj > 0, f"[{size}] proj_in grad zero (path not wired)"
    assert max_wo > 0, f"[{size}] W_o grad zero at init"
    assert max_gate > 0, f"[{size}] dil_gate grad zero (content gate not wired)"
    print(f"  max conv={max_conv:.2e}  max proj_in={max_proj:.2e}  "
          f"max W_o={max_wo:.2e}  max dil_gate={max_gate:.2e}")
    print(f"  (log_tau grad may be 0 while dil_gate logits are 0 -- expected)")
    print(f"  [T2 PASS]")
    passes += 1

    # === Causality at every lag (all branches left-pad (k-1)*d) ===
    # Use S=256 for presets with k <= 129 (pad up to (129-1)*8 = 1024 at dil=8
    # would need S > 1024; we keep S=256 and verify structural causality at
    # 3 positions). For tiny presets the receptive field is smaller, making
    # the test stricter (more positions fully out of the conv's reach).
    print(f"\n[C-{size}] Causality at every lag (S=256, perturb at 130/200/255):")
    m4 = MicroMixerV87Final(cfg)
    m4.eval()
    S_t = 256
    ids_a = torch.randint(4, V, (1, S_t))
    perturb_positions = [130, 200, 255]
    max_diff = 0.0
    for p in perturb_positions:
        ids_b = ids_a.clone()
        ids_b[0, p] = (ids_b[0, p] + 1) % V
        with torch.no_grad():
            la = m4(ids_a, None, None, None, None)[0]
            lb = m4(ids_b, None, None, None, None)[0]
        d = (la[:, :p] - lb[:, :p]).abs().max().item()
        max_diff = max(max_diff, d)
        print(f"  perturb pos {p:>3d}/{S_t}: max past-logit diff (< {p}) = {d:.2e}")
    assert max_diff == 0.0, f"[{size}] causality violated: {max_diff}"
    print(f"  overall max past-logit diff = {max_diff:.2e}  "
          f"[PASS] (branches causal, gate pointwise)")
    passes += 1

    return passes


# Preset function lookup (defined here to avoid forward-reference issues in
# the self-test dispatcher).
_PRESET_FNS = {
    "1M":   v87_final_1m,
    "500K": v87_final_500k,
    "300K": v87_final_300k,
    "100K": v87_final_100k,
    "50K":  v87_final_50k,
    "10K":  v87_final_10k,
}


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 72)
    print("MicroMixer-4 V87 Final -- self-test")
    print("(V83-RPG as the SINGLE architecture, scaled to 6 budgets)")
    print("(Content-Controlled Dilation: non-LTI token-mix, per-position "
          "dilation gate, V76 full-RoPE on v AND g)")
    print(f"(6 presets: {' '.join(_V87_PRESETS)})")
    print("=" * 72)

    V = 256
    ALL_SIZES = list(_V87_PRESETS)
    PER_PRESET_TESTS = 3  # T1, T2, Causality
    CROSS_PRESET_TESTS = 4  # Interface, Module scan, Generation+deepcopy, CCD-equiv

    TOTAL = len(ALL_SIZES) * PER_PRESET_TESTS + CROSS_PRESET_TESTS
    PASS_COUNT = 0

    # === Per-preset tests ===
    for size in ALL_SIZES:
        try:
            passes = _run_preset_tests(size, V)
            PASS_COUNT += passes
        except AssertionError as e:
            print(f"\n  *** FAIL: {e}")
            raise

    # === Cross-preset tests ===
    print(f"\n{'='*72}")
    print("  CROSS-PRESET TESTS")
    print(f"{'='*72}")

    # === [INT] Interface contract (on 1M preset) ===
    print("\n[INT] Interface contract (forward sig, generate kwargs, aux, "
          "tied head, deepcopy, strict-load, state_dict key counts):")
    import inspect
    sig = inspect.signature(MicroMixerV87Final.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "input_ids", "q_mask", "a_mask", "all_mask", "targets"], \
        f"forward signature mismatch: {params}"
    gsig = inspect.signature(MicroMixerV87Final.generate)
    gparams = list(gsig.parameters.keys())
    assert gparams == ["self", "input_ids", "q_mask", "a_mask", "all_mask",
                       "max_new_tokens", "temperature", "top_k", "top_p",
                       "repetition_penalty", "no_repeat_ngram_size", "eos_token_id"], \
        f"generate signature mismatch: {gparams}"
    gdefs = {p: gsig.parameters[p].default for p in gparams if p != "self"}
    assert gdefs["max_new_tokens"] == 100
    assert gdefs["temperature"] == 1.0
    assert gdefs["top_k"] is None
    assert gdefs["top_p"] is None
    assert gdefs["repetition_penalty"] == 1.0
    assert gdefs["no_repeat_ngram_size"] == 0
    assert gdefs["eos_token_id"] == 2

    # aux keys + config type + embed type + tied head (on 1M preset).
    m_int = MicroMixerV87Final(v87_final_1m())
    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat(
        [ids[:, 1:], torch.full((B, 1), 0, dtype=torch.long)], dim=1,
    )
    _, _, aux_full = m_int(ids, None, None, None, targets)
    expected_aux = {"ce_loss", "ce_unweighted", "main_loss", "token_acc", "loss_weight_mean"}
    assert expected_aux.issubset(aux_full.keys()), f"aux missing keys: {aux_full.keys()}"
    _, _, aux_none = m_int(ids, None, None, None, None)
    assert set(aux_none.keys()) == {"token_acc"}, f"aux(targets=None) wrong: {aux_none.keys()}"
    assert isinstance(m_int.out_norm, RMSNorm), "out_norm not RMSNorm"
    assert isinstance(m_int.config, V87Config), "config not V87Config"
    assert isinstance(m_int.config, V71Config), "config not V71Config (subclass)"
    assert isinstance(m_int.embed, nn.Embedding), "embed not nn.Embedding"
    assert m_int.config.d_model == 128
    assert m_int.config.max_seq_len == 1024
    assert m_int.config.pad_token_id == 0
    # Tied head: NO standalone output Linear; logits come from embed.weight.T.
    has_separate_head = any(
        isinstance(mod, nn.Linear) and mod.out_features == V
        and n.endswith("lm_head")
        for n, mod in m_int.named_modules()
    )
    assert not has_separate_head, "found standalone lm_head (head must be tied)"

    # deepcopy on a FRESH 1M model.
    m_fresh = MicroMixerV87Final(v87_final_1m())
    m_fresh.eval()
    m_deep = copy.deepcopy(m_fresh)
    m_deep.eval()
    test_ids2 = torch.randint(4, V, (2, 32))
    with torch.no_grad():
        out_orig = m_fresh(test_ids2, None, None, None, None)[0]
        out_deep = m_deep(test_ids2, None, None, None, None)[0]
    deep_diff = (out_orig - out_deep).abs().max().item()
    assert deep_diff == 0.0, f"deepcopy not bit-identical: {deep_diff}"

    # Strict state_dict round-trip on 1M preset.
    m_strict_src = MicroMixerV87Final(v87_final_1m())
    m_strict_dst = MicroMixerV87Final(v87_final_1m())
    sd = m_strict_src.state_dict()
    missing, unexpected = m_strict_dst.load_state_dict(sd, strict=True)
    assert not missing, f"strict-load missing keys: {missing}"
    assert not unexpected, f"strict-load unexpected keys: {unexpected}"
    m_strict_src.eval(); m_strict_dst.eval()
    with torch.no_grad():
        out_src = m_strict_src(test_ids2, None, None, None, None)[0]
        out_dst = m_strict_dst(test_ids2, None, None, None, None)[0]
    load_diff = (out_src - out_dst).abs().max().item()
    assert load_diff == 0.0, f"strict-load not bit-identical: {load_diff}"

    # state_dict key counts per preset (formula: 3 + 27 * num_blocks).
    for size in ALL_SIZES:
        sd_s = MicroMixerV87Final(_PRESET_FNS[size]()).state_dict()
        _, N_s, _ = _V87_PRESET_SPECS[size]
        exp_keys = _expected_sd_keys(N_s)
        assert len(sd_s) == exp_keys, (
            f"[{size}] expected {exp_keys} state_dict keys, got {len(sd_s)}"
        )

    print(f"  forward 5 args OK; aux keys = {sorted(aux_full.keys())}")
    print(f"  generate kwargs defaults OK (V76 parity)")
    print(f"  tied head (no standalone lm_head) OK")
    print(f"  deepcopy(1M, fresh) bit-identical (diff={deep_diff:.0e})")
    print(f"  strict state_dict round-trip bit-identical (diff={load_diff:.0e})")
    print(f"  state_dict keys per preset: "
          f"{ {s: _expected_sd_keys(_V87_PRESET_SPECS[s][1]) for s in ALL_SIZES} }")
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === [SCAN] Module scan: no attention / recurrence / SSM; structural counts ===
    print("\n[SCAN] Module scan (no attention/recurrence/SSM; "
          "GLCTokenMixCCD/SwiGLU/ReMixerLayer == num_blocks per preset):")
    m_scan = MicroMixerV87Final(v87_final_1m())
    forbidden_types = (nn.MultiheadAttention, nn.RNN, nn.LSTM, nn.GRU,
                       nn.RNNCell, nn.LSTMCell, nn.GRUCell)
    has_forbidden_type = any(
        isinstance(mod, forbidden_types) for _, mod in m_scan.named_modules()
    )
    has_forbidden_name = any(
        ("attention" in n.lower() or "attn" in n.lower()
         or "s4" in n.lower() or "mamba" in n.lower())
        for n, _ in m_scan.named_modules()
    )
    assert not has_forbidden_type, "forbidden module type present"
    assert not has_forbidden_name, "forbidden module name present"
    # Count structural modules per preset.
    for size in ALL_SIZES:
        m_s = MicroMixerV87Final(_PRESET_FNS[size]())
        _, N_s, _ = _V87_PRESET_SPECS[size]
        n_tokenmix = sum(
            1 for _, mod in m_s.named_modules() if isinstance(mod, GLCTokenMixCCD)
        )
        n_swiglu = sum(
            1 for _, mod in m_s.named_modules() if isinstance(mod, SwiGLUChannelMLP)
        )
        n_remixer = sum(
            1 for _, mod in m_s.named_modules() if isinstance(mod, ReMixerLayer)
        )
        assert n_tokenmix == N_s, (
            f"[{size}] expected {N_s} GLCTokenMixCCD, got {n_tokenmix}"
        )
        assert n_swiglu == N_s, (
            f"[{size}] expected {N_s} SwiGLU, got {n_swiglu}"
        )
        assert n_remixer == N_s, (
            f"[{size}] expected {N_s} ReMixerLayer, got {n_remixer}"
        )
    print(f"  forbidden types/names: {has_forbidden_type or has_forbidden_name}")
    print(f"  GLCTokenMixCCD / SwiGLU / ReMixerLayer per preset: "
          f"all == num_blocks "
          f"({ {s: _V87_PRESET_SPECS[s][1] for s in ALL_SIZES} })")
    print(f"  [PASS] no attention/recurrence/SSM")
    PASS_COUNT += 1

    # === [GEN] Generation smoke + RoPE buffer check (on 1M preset) ===
    print("\n[GEN] Generation smoke + RoPE buffer check:")
    m_gen = MicroMixerV87Final(v87_final_1m())
    m_gen.eval()
    prompt = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)
    out_greedy = m_gen.generate(prompt, max_new_tokens=20, temperature=0.0,
                                eos_token_id=-1)
    assert out_greedy.shape == (1, 25), f"greedy shape {out_greedy.shape}"
    assert torch.equal(out_greedy[:, :5], prompt), "prompt corrupted"
    assert (out_greedy[:, 5:] >= 0).all() and (out_greedy[:, 5:] < V).all(), \
        "bad byte range"
    out_sample = m_gen.generate(prompt, max_new_tokens=16, temperature=1.0,
                                eos_token_id=-1)
    assert out_sample.shape == (1, 21), f"sample shape {out_sample.shape}"
    # RoPE buffers non-persistent (absent from state_dict, present as buffers).
    sd = m_gen.state_dict()
    assert "rope_cos" not in sd and "rope_sin" not in sd, \
        "RoPE gate buffers leaked into state_dict"
    assert "rope_cos_state" not in sd and "rope_sin_state" not in sd, \
        "RoPE state buffers leaked into state_dict"
    buf_names = dict(m_gen.named_buffers())
    has_rope_gate = any(n.endswith("token_mix.rope_cos") for n in buf_names)
    has_rope_state = any(n.endswith("token_mix.rope_cos_state") for n in buf_names)
    assert has_rope_gate, "RoPE gate buffers missing"
    assert has_rope_state, "RoPE state buffers missing"
    print(f"  greedy: shape {tuple(out_greedy.shape)}, prompt preserved")
    print(f"  T=1.0:  shape {tuple(out_sample.shape)}, prompt preserved")
    print(f"  RoPE gate buffers:  present as non-persistent ({has_rope_gate})")
    print(f"  RoPE state buffers: present as non-persistent ({has_rope_state})")
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === [CCD] Perfect-ablation equivalence (1M preset vs V76-RPG parent) ===
    # (a) V87-1M is a superset of V76-RPG state_dict keys: the 21 CCD gate keys
    #     (7 blocks x [dil_gate.weight, dil_gate.bias, log_tau]) are the ONLY
    #     extras; no V76 keys are missing.
    # (b) V76-RPG state_dict loads into V87-1M with strict=False: missing keys
    #     are exactly the 21 CCD gate keys; no unexpected keys.
    # (c) With matched (un-zeroed) W_o and force_dilation=0 (one-hot dil=1),
    #     V87-1M forward is BIT-IDENTICAL to V76-RPG forward; with the gate
    #     free (uniform at init) the forwards DIFFER by > 1e-3 (mechanism
    #     active). This is the V83-RPG perfect-ablation property.
    print("\n[CCD] Perfect-ablation equivalence (force_dilation=0 => V76 parent; "
          "free gate => mechanism active):")
    m_v76 = MicroMixerV76RPG(v76_rpg_1m())
    m_v87 = MicroMixerV87Final(v87_final_1m())

    sd_v76 = m_v76.state_dict()
    sd_v87 = m_v87.state_dict()
    ccd_suffixes = ("dil_gate.weight", "dil_gate.bias", "log_tau")
    missing_87 = [k for k in sd_v76.keys() if k not in sd_v87]
    extra_87 = [k for k in sd_v87.keys() if k not in sd_v76]
    assert not missing_87, f"V76 keys missing from V87: {missing_87}"
    assert len(extra_87) == 21 and all(
        k.endswith(ccd_suffixes) for k in extra_87
    ), f"unexpected V87-only keys: {sorted(extra_87)}"
    print(f"  V87-1M superset of V76-RPG keys: +21 CCD keys exactly [OK]")

    missing_l, unexpected_l = m_v87.load_state_dict(sd_v76, strict=False)
    assert not unexpected_l, f"unexpected keys loading V76 into V87: {unexpected_l}"
    assert len(missing_l) == 21, f"expected 21 missing keys, got {len(missing_l)}"
    print(f"  V87.load_state_dict(V76 sd, strict=False): missing=21 CCD keys [OK]")

    # Un-zero W_o IDENTICALLY in both models.
    torch.manual_seed(7)
    for b in m_v76.blocks:
        nn.init.xavier_uniform_(b.token_mix.W_o.weight)
        nn.init.zeros_(b.token_mix.W_o.bias)
    for b76, b87 in zip(m_v76.blocks, m_v87.blocks):
        b87.token_mix.W_o.weight.data.copy_(b76.token_mix.W_o.weight.data)
        b87.token_mix.W_o.bias.data.copy_(b76.token_mix.W_o.bias.data)
    m_v76.eval(); m_v87.eval()
    chk_ids = torch.randint(4, V, (2, 128))

    _set_force_dilation(m_v87, 0)
    with torch.no_grad():
        out_v76 = m_v76(chk_ids, None, None, None, None)[0]
        out_v87_f0 = m_v87(chk_ids, None, None, None, None)[0]
    equiv_diff = (out_v76 - out_v87_f0).abs().max().item()
    print(f"  max |V76-RPG - V87(force=0)| = {equiv_diff:.2e} "
          f"(must be 0.0 -- perfect ablation)")
    assert equiv_diff == 0.0, (
        f"V87 with force_dilation=0 is NOT bit-identical to V76-RPG: "
        f"{equiv_diff}"
    )

    _set_force_dilation(m_v87, None)
    with torch.no_grad():
        out_v87_free = m_v87(chk_ids, None, None, None, None)[0]
    mech_diff = (out_v76 - out_v87_free).abs().max().item()
    print(f"  max |V76-RPG - V87(free gate)| = {mech_diff:.4e} "
          f"(must be > 1e-3 -- CCD mechanism active)")
    assert mech_diff > 1e-3, (
        f"CCD free-gate forward does not differ from parent: {mech_diff}"
    )
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === Final summary ===
    print("\n" + "=" * 72)
    print(f"V87 Final self-test: {PASS_COUNT}/{TOTAL} PASS")
    print("=" * 72)
    if PASS_COUNT == TOTAL:
        print("ALL V87 SELF-TESTS PASSED")
    else:
        print(f"FAIL: {TOTAL - PASS_COUNT} test(s) did not pass")
        raise SystemExit(1)

    print("\nFinal per-preset param counts:")
    for size in ALL_SIZES:
        c = _PRESET_FNS[size]()
        n = count_parameters(MicroMixerV87Final(c))
        d, N, k = _V87_PRESET_SPECS[size]
        print(f"  v87_final_{size.lower():>4s}: {n:>10,} params  "
              f"(d={d}, blocks={N}, glc_kernel={k}, "
              f"dilations={tuple(c.ccd_dilations)}, "
              f"budget={_V87_BUDGETS[size]:,}, "
              f"{n/_V87_BUDGETS[size]*100:.1f}%)")
