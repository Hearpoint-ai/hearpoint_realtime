# Deep Dive: Why "Look Back" Degrades Target Speech Extraction

## Context

When demoing the real-time TSE system with binaural mics, enrollment works great while looking at the target speaker. Looking away (90 degrees) causes extraction to stop (expected per the paper -- HRTF changes too drastically). But when looking **back**, extraction resumes at **worse quality** than the initial session. This document analyzes why.

The "not working when looking away" is expected. The question is specifically: **why does looking back not restore full quality?**

---

## Root Cause Analysis: Hypothesis Ranking

### #1 (PRIMARY, ~85%): Inter-Frame LSTM State Poisoning

**The smoking gun.** Each of the 3 GridNet blocks has a causal inter-frame LSTM (`H=64`) whose hidden state (`h0`) and cell state (`c0`) carry across every 8ms chunk indefinitely.

**Mechanism:**
1. During enrollment + initial listening, LSTM states encode on-axis binaural temporal dynamics (ITD, ILD, spectral cues at that angle)
2. When you look away, HRTF changes drastically. Over hundreds of chunks (~seconds), the LSTM cell states deeply integrate off-axis characteristics
3. When you look back, those states are NOT reset -- no trigger exists for orientation changes
4. LSTM cell states have **long memory by design** (forget gate controls decay). The off-axis contamination must "wash out" over many chunks
5. With 3 blocks cascading (block 0 output feeds block 1's embedding gate, feeds block 2), the corruption compounds

**Key code** (`tfgridnet_causal.py:540-546`):
```python
inter_rnn, (h0, c0) = self.inter_rnn(inter_rnn, (h0, c0))  # state carried forever
init_state['h0'] = h0  # no decay, no reset, no gating
init_state['c0'] = c0
```

**State resets only happen on:** embedding change, passthrough toggle, or input queue overflow (>8 chunks). Head movement triggers **nothing**.

### #2 (SECONDARY, ~65%): Attention K/V Buffer Contamination

Each block's causal self-attention holds 49 frames (~392ms) of K/V history. When looking back, the first ~49 on-axis chunks attend against off-axis K/V context, producing confused attention patterns.

**Self-correcting in exactly 392ms** as old entries flush out. This explains the *initial* degradation burst on look-back but NOT sustained degradation. If quality stays bad past ~400ms, this isn't the cause -- H1 is.

**Compounds H1**: During the 392ms flush window, corrupted attention output feeds into the LSTM, further delaying LSTM washout.

### #3 (CONTRIBUTING, ~50%): Embedding-HRTF Angle Mismatch

The 256-dim speaker embedding was computed from audio at a **specific head angle**. When you look back, your head may not be at the *exact* same angle. Even 5-10 degrees difference changes the HRTF, and since the embedding captures binaural spatial info (especially with the beamformer enrollment path), the multiplicative gate (`batch = batch * embed` at layer 1) applies slightly wrong frequency-wise scaling.

**This is persistent** -- doesn't wash out because the embedding is static. Explains mild sustained degradation even after LSTM states recover.

### #4 (MINOR, ~15%): Overlap-Add Buffer Artifacts

`conv_buf` and `deconv_buf` hold only 2 frames (16ms), `istft_buf` holds 1 frame. Contamination flushes in <24ms -- imperceptible. Real but negligible.

### #5 (CONFOUND, ~25%): Underrun-Related Degradation

File 2 (look-back scenario) shows 134 underruns vs 92 in File 1 (+45%). However:
- File 2 is longer (62s vs 53s), so the *rate* may be similar
- Underruns fill output with zeros (audible dropouts) but don't explain model quality degradation
- If underruns trigger queue drain (>8 chunks), that actually **resets** state, which would *help* not hurt
- More likely a confound than a cause

---

## Log Analysis Summary

| Metric | File 1 (look -> away) | File 2 (look -> away -> back) |
|--------|----------------------|------------------------------|
| RTF avg (isolation) | 0.6691 | 0.6711 |
| Output drops | 34 | 18 |
| Underruns | 92 | 134 |
| Max latency | 81.1ms | 63.8ms |

**Missing from logs:** No per-chunk embedding similarity scores, no spatial/beamforming metrics, no state reset indicators. The current perf logger doesn't capture the metrics needed to directly observe the degradation mechanism.

---

## Proposed Experiments (ordered by priority)

### Exp 1: Manual LSTM State Reset on Look-Back (IMPLEMENTED)

Added 'R' keybind in the demo that resets ONLY LSTM h0/c0 across all blocks, preserving everything else:

```python
# engine.py
def reset_lstm_states_only(self):
    for key in self.state['gridnet_bufs']:
        self.state['gridnet_bufs'][key]['h0'].zero_()
        self.state['gridnet_bufs'][key]['c0'].zero_()
```

**If pressing R when looking back restores full quality --> H1 confirmed.**

### Exp 2: Full State Reset on Look-Back

Use existing passthrough toggle (off then on) to trigger `_reset_runtime_context()` (full reset of all buffers). Compare quality to Exp 1 to isolate LSTM contribution vs attention/overlap buffers.

### Exp 3: Embedding Angle Sensitivity

Record enrollment audio at 0, 5, 10, 15, 20 degree offsets. Compute cosine similarity between embeddings. If similarity drops rapidly with angle, H3 is significant.

### Exp 4: Per-Snapshot Quality Logging (IMPLEMENTED)

Added per-snapshot cosine similarity logging between enrollment embedding and a re-embedded version of recent output audio. This lets us observe the recovery curve after look-back and measure washout time.

---

## Proposed Mitigations

### M1 (Quick win): State Decay Factor
Apply exponential decay to LSTM states each chunk:
```python
decay = 0.995  # tunable -- limits max memory depth
init_state['h0'] = h0 * decay
init_state['c0'] = c0 * decay
```
Ensures off-axis contamination fades even without explicit reset. Must tune: too aggressive kills temporal modeling.

### M2 (Quick win): Manual/Automatic State Reset
- Add a "reset" keybind to the demo for quick recovery (DONE - 'R' key)
- Or: detect when output energy drops below threshold (model suppressing everything) and auto-reset

### M3 (Medium-term): IMU-Triggered State Reset
If hardware has an IMU, trigger `_reset_runtime_context()` when head orientation changes beyond a threshold (e.g., >30 degrees). Directly prevents the problem.

### M4 (Longer-term): Angle-Invariant Embeddings
- Average embeddings from multiple head angles during enrollment
- Or train embedding model to be angle-invariant (speaker identity only, strip spatial info)

---

## Architecture Reference

### Stateful Components per GridNet Block (3 blocks total)

| Buffer | Shape | Memory Depth | Flush Time |
|--------|-------|-------------|------------|
| h0 (LSTM hidden) | [1, 97, 64] | Unbounded (forget-gated) | Seconds+ |
| c0 (LSTM cell) | [1, 97, 64] | Unbounded (forget-gated) | Seconds+ |
| K_buf (attention) | [4, 49, E*97] | 49 frames | 392ms |
| V_buf (attention) | [4, 49, 16*97] | 49 frames | 392ms |

### Top-Level Buffers

| Buffer | Shape | Memory Depth | Flush Time |
|--------|-------|-------------|------------|
| conv_buf | [1, 4, 2, 97] | 2 frames | 16ms |
| deconv_buf | [1, 64, 2, 97] | 2 frames | 16ms |
| istft_buf | [1, n_srcs, 194, 1] | 1 frame | 8ms |

### Speaker Embedding Conditioning
- Static 256-dim vector projected to [64 x 97] = [6208] dimensions
- Applied as multiplicative gate: `batch = batch * embed` at block 1 only
- Computed once at enrollment, never updated

### State Reset Triggers
1. `set_embedding` -> full `_reset_runtime_context()`
2. Passthrough toggle -> full `_reset_runtime_context()`
3. Input queue overflow (>8 chunks) -> drain + `init_buffers()`
4. **NEW: 'R' key** -> `reset_lstm_states_only()` (LSTM h0/c0 only)

---

## Critical Files

| File | Relevance |
|------|-----------|
| `src/models/tfgridnet_realtime/tfgridnet_causal.py` | LSTM state flow (L540-546), attention K/V buffers (L567-576), init_buffers (L422-441) |
| `src/realtime/engine.py` | State reset logic (`_reset_runtime_context` L258), drain threshold (L641-654), `_process_chunk` (L417+) |
| `src/models/tfgridnet_realtime/net.py` | Model wrapper, `init_buffers()` and `predict()` interface |
| `scripts/demo.py` | Demo UI with keybinds |
| `src/realtime/perf_logger.py` | Performance and quality metric logging |
| `src/configs/tfgridnet_cipic.json` | Model config: B=3, H=64, local_atten_len=50, embed_dim=256 |
