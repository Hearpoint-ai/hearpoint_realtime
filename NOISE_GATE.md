# Noise Gate & Auto-Reset System

This document describes the post-model audio processing pipeline added to suppress interferer leakage and recover from LSTM state poisoning during real-time speaker isolation.

## Problem

TFGridNet uses a speaker embedding to identify **who** to isolate, but it needs the target speaker's voice to be **present** in the mix to do its job. This creates two failure modes:

1. **Interferer leakage** -- When the target speaker is silent, the model has nothing to extract. Interfering speakers leak through the output because the model defaults to passing audio through.

2. **State poisoning** -- When the user looks away from the target speaker (changing the microphone array response), the model's LSTM state accumulates bad temporal patterns. The output fades even when the target IS speaking. Previously this required a manual full reset (F key).

## Architecture

The processing pipeline in isolation mode:

```
Input audio
  -> input_gain (boost before model)
  -> MODEL INFERENCE (stateful, with speaker embedding)
  -> spectral subtraction (stationary noise removal)
  -> mono mix (if configured)
  -> [capture pre-gate output level]
  -> NOISE GATE (suppress interferer leakage)
  -> AUTO-RESET CHECK (detect state poisoning, uses pre-gate levels)
  -> output_gain (volume control)
  -> clip to [-1, 1]
  -> Output audio
```

The noise gate and auto-reset are independent systems that address different problems. The auto-reset deliberately uses **pre-gate** output levels so the gate's attenuation doesn't trigger false resets.

---

## Noise Gate

### What it does

Suppresses model output when the target speaker isn't talking. When the model output energy drops below a threshold for a sustained period, the gate smoothly ramps the output to silence. When the target starts speaking again, the gate opens quickly.

### State machine

The gate cycles through five states:

```
CLOSED  ->  ATTACK  ->  OPEN  ->  HOLD  ->  RELEASE  ->  CLOSED
         (ramp up)            (bridge gaps)  (ramp down)
```

- **CLOSED** -- Output is muted (gain = 0). Transitions to ATTACK when the smoothed envelope rises above the energy threshold.
- **ATTACK** -- Output ramps up linearly over `attack_chunks`. If energy drops below threshold during attack, returns to CLOSED immediately.
- **OPEN** -- Full output (gain = 1). Transitions to HOLD when energy drops below threshold.
- **HOLD** -- Full output maintained for `hold_chunks` to bridge gaps between words. If energy returns, goes back to OPEN. If hold expires, transitions to RELEASE.
- **RELEASE** -- Output ramps down linearly over `release_chunks`. If energy returns during release, jumps to ATTACK. If release completes, transitions to CLOSED.

### Envelope smoothing

Raw per-chunk energy (peak absolute value) is smoothed with an exponential moving average to prevent single-sample spikes from opening the gate:

```
envelope = smooth_coeff * previous_envelope + (1 - smooth_coeff) * current_energy
```

### Configuration

In `config.yaml` under `noise_gate:`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable the noise gate |
| `energy_threshold` | `0.005` | Output peak amplitude below which the gate closes |
| `attack_chunks` | `2` | Chunks to ramp open (~16ms at 128 samples/16kHz) |
| `hold_chunks` | `15` | Chunks to stay open after energy drops (~120ms) |
| `release_chunks` | `10` | Chunks to ramp closed (~80ms) |
| `smooth_coeff` | `0.3` | Envelope smoothing (0 = instant, 1 = frozen) |

### Tuning

Press **V** in the demo to toggle verbose gate diagnostics. The display shows:

| Field | Meaning |
|-------|---------|
| **OutEnergy** | Raw model output peak amplitude (pre-gate, pre-output-gain) |
| **Envelope** | Smoothed energy envelope (compared against threshold) |
| **Threshold** | Current `energy_threshold` from config |
| **Gate** | Current state (OPEN/HOLD/ATTACK/RELEASE/CLOSED), color-coded |
| **GateGain** | Current gain multiplier (0.0 to 1.0) |
| **InLevel** | Input audio level (pre-input-gain) |
| **IO Ratio** | OutEnergy / InLevel -- how much the model is passing through |

**Tuning workflow:**

1. Run `make demo`, enroll a target speaker, press V
2. Have the target speaker talk -- note the **OutEnergy** values (this is your "signal" level)
3. Have the target go silent with interferers present -- note the **OutEnergy** values (this is your "leak" level)
4. Set `energy_threshold` between the two values
5. Adjust `hold_chunks` if words get chopped (increase) or interferers bleed between words (decrease)

---

## Auto-Reset Detector

### What it does

Detects when the model's internal state (LSTM hidden/cell states, attention buffers, conv/deconv context) has been poisoned -- typically from head orientation changes that alter the microphone array response. When detected, it automatically triggers a full state reset (equivalent to pressing F).

### Detection algorithm

Each chunk, the detector compares input and output energy:

1. **Skip if cooldown is active** -- after a reset, wait `cooldown_chunks` before checking again
2. **Skip if input is quiet** -- if input level < `input_floor`, there's no audio to evaluate; reset the suspect counter
3. **Compute ratio** -- `output_level / input_level`
4. **Evaluate** -- if ratio < `ratio_threshold`, increment the suspect counter; otherwise reset it to 0
5. **Trigger** -- if the suspect counter reaches `consecutive_chunks`, fire a full reset and start the cooldown

The key insight: state poisoning manifests as the model suppressing output even when meaningful input is present. A healthy model in isolation mode preserves the target speaker at a significant fraction of input energy. A poisoned model drives output near zero.

### Configuration

In `config.yaml` under `auto_reset:`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable auto-reset |
| `input_floor` | `0.005` | Minimum input level to consider "active audio" |
| `ratio_threshold` | `0.05` | Output/input ratio below which state is suspect |
| `consecutive_chunks` | `50` | Sustained suspect chunks before reset (~400ms) |
| `cooldown_chunks` | `250` | Chunks to wait after reset before checking again (~2s) |

### Interaction with noise gate

The auto-reset uses **pre-gate** output levels. This is critical: the noise gate legitimately zeros the output when the target is silent, which would look like state poisoning to the detector. By capturing the model's raw output level before the gate modifies it, the two systems operate independently without interfering.

### CLI feedback

When an auto-reset fires, the demo status bar displays:

```
Status: [auto-reset] State reset #N triggered
```

---

## Input Gain

A simple linear multiplier applied to input audio **before** model inference. Useful when the microphone input is too quiet for the model to work effectively.

Configured in `config.yaml` under `audio:`:

```yaml
audio:
  input_gain: 4.0  # 1.0 = unity, 2.0 = +6dB
```

This is independent of `output_gain`, which controls the final output volume after all processing.
