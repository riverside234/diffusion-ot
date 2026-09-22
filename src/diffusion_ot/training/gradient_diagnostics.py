"""Read-only, sampled objective-gradient diagnostics for Stage 1B.

These measure weighted gradients before clipping/Adam, not parameter updates.
Negative cosine is evidence of conflict, not proof that PCGrad improves images.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math

import torch


def objective_gradients(loss, parameters):
    if not parameters or not loss.requires_grad:
        return (None,) * len(parameters)
    return tuple(None if g is None else g.detach() for g in torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True))


def combine_gradients(*terms):
    """Linear combination without modifying its detached input gradients."""
    if len({len(gradients) for _, gradients in terms}) != 1:
        raise ValueError("Gradient lists must have matching lengths.")
    result = []
    for entries in zip(*(gradients for _, gradients in terms)):
        live = [scale * g for (scale, _), g in zip(terms, entries) if g is not None and scale != 0]
        result.append(sum(live[1:], live[0]) if live else None)
    return tuple(result)


def gradient_pair_metrics(first, second, *, eps=1e-12):
    """Treat unused parameters as zero; zero-norm pairs have undefined cosine.

    ``*_pair_descent_fraction`` is g_i dot (g_first + g_second) / ||g_i||^2.
    A negative value predicts an increase of that loss for an infinitesimal
    negative-gradient step using ONLY this pair. It is not an Adam prediction.
    """
    if len(first) != len(second):
        raise ValueError("Gradient lists must have matching lengths.")
    by_device = {}
    for a, b in zip(first, second):
        if a is None and b is None:
            continue
        device = (a if a is not None else b).device
        stats = by_device.setdefault(device, torch.zeros(3, device=device, dtype=torch.float32))
        if a is not None:
            stats[0] += a.detach().float().square().sum()
        if b is not None:
            stats[1] += b.detach().float().square().sum()
        if a is not None and b is not None:
            if a.shape != b.shape or a.device != b.device:
                raise ValueError("Corresponding gradients must have matching shapes and devices.")
            stats[2] += (a.detach().float() * b.detach().float()).sum()
    values = [s.cpu().tolist() for s in by_device.values()]
    aa, bb, dot = (sum(v[i] for v in values) for i in range(3))
    if not all(math.isfinite(v) for v in (aa, bb, dot)):
        raise FloatingPointError("Non-finite objective-gradient diagnostic.")
    a, b = math.sqrt(aa), math.sqrt(bb)
    valid = a > eps and b > eps
    cosine = max(-1., min(1., dot / (a * b))) if valid else None
    return {
        "valid": valid, "first_norm": a, "second_norm": b, "dot": dot,
        "cosine": cosine, "conflict": dot < 0 if valid else None,
        "first_to_second_norm_ratio": a / b if b > eps else None,
        # Single pairwise PCGrad projection removes this fraction of either
        # gradient's norm; this is a diagnostic, not an applied projection.
        "projection_removed_norm_fraction": max(0., -cosine) if valid else None,
        "first_pair_descent_fraction": (aa + dot) / aa if a > eps else None,
        "second_pair_descent_fraction": (bb + dot) / bb if b > eps else None,
    }


class GradientConflictMonitor:
    """Last 20 probes within a process/phase; no optimizer/RNG state."""
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        self.phase = "unspecified"

    def set_phase(self, phase):
        if phase != self.phase:
            self.history.clear()
            self.phase = phase

    def record(self, group, pair, metrics):
        history = self.history[(group, pair)]
        history.append(metrics["cosine"])
        valid = [value for value in history if value is not None]
        return {**metrics, "window_probes": len(history), "window_valid_pairs": len(valid),
                "window_conflict_rate": sum(v < 0 for v in valid) / len(valid) if valid else None,
                "window_mean_cosine": sum(valid) / len(valid) if valid else None}


def training_gradient_conflicts(losses, groups, *, code_gradients, encoder_scale, monitor):
    """Collect each objective once; report domain-specific and joint groups.

    Groups are disjoint ``encoder.cat``, ``generator.dog``, etc. Code gradients
    are precomputed by selective autograd and indexed by parameter identity,
    so the readout/conditioning encoder is never assigned a code gradient.
    """
    parameters = [p for values in groups.values() for p in values]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Diagnostic parameter groups must not overlap.")
    collected = {name: objective_gradients(loss, parameters) for name, loss in losses.items()}
    collected["code"] = tuple(code_gradients.get(id(p)) for p in parameters)
    indices = {}
    start = 0
    for name, values in groups.items():
        indices[name] = list(range(start, start + len(values)))
        start += len(values)
    for kind in dict.fromkeys(name.split(".")[0] for name in groups):
        indices[f"{kind}.all"] = [i for name, ids in indices.items()
                                  if name.startswith(kind + ".") for i in ids]
    report, norms = {}, {}
    for name, ids in indices.items():
        kind = name.split(".")[0]
        grads = {key: tuple(value[i] for i in ids) for key, value in collected.items()}
        zero = (None,) * len(ids)
        for objective, values in grads.items():
            norms.setdefault(objective, {})[name] = gradient_pair_metrics(values, zero)["first_norm"]
        scale = encoder_scale if kind == "encoder" else 1.
        decoded = combine_gradients((scale, grads["decoded"]))
        perceptual = combine_gradients((scale, grads["perceptual"]))
        adversarial = combine_gradients((scale, grads["adversarial"]))
        color = combine_gradients((scale, grads.get("color", zero)))
        # An explicitly disabled structure objective must stay exactly zero;
        # subtracting large gradient sums can leave spurious roundoff conflicts.
        structure = (combine_gradients((scale, grads["structure"])) if "structure" in grads else
                     combine_gradients((1., decoded), (-1., adversarial), (-1., color), (-1., perceptual)))
        dino = combine_gradients((1., perceptual), (1., structure))
        translation = combine_gradients((1., decoded), (1., grads["code"]))
        pairs = {
            "dino_vs_adversarial": (dino, adversarial),
            "perceptual_vs_adversarial": (perceptual, adversarial),
            "perceptual_vs_structure": (perceptual, structure),
        }
        if "color" in grads:
            pairs["color_vs_adversarial"] = (color, adversarial)
            pairs["color_vs_perceptual"] = (color, perceptual)
            if kind != "matching_head":
                pairs["color_vs_reconstruction"] = (color, grads["reconstruction"])
        if kind != "matching_head":
            pairs["translation_vs_reconstruction"] = (translation, grads["reconstruction"])
        if kind != "generator":
            pairs["matching_vs_translation"] = (grads["matching"], translation)
        if kind == "encoder":
            pairs["matching_vs_reconstruction"] = (grads["matching"], grads["reconstruction"])
        if kind != "generator" and "conditional" in grads:
            # Isolate KL: comparing it with the aggregate matching objective
            # would include its own gradient and bias the cosine upward.
            pairs["conditional_vs_translation"] = (grads["conditional"], translation)
            if kind == "encoder":
                pairs["conditional_vs_reconstruction"] = (grads["conditional"], grads["reconstruction"])
            for component in ("infoot", "protection"):
                if component in grads:
                    pairs[f"conditional_vs_{component}"] = (grads["conditional"], grads[component])
        if kind == "generator":
            pairs["code_vs_decoded"] = (grads["code"], decoded)
            pairs["code_vs_reconstruction"] = (grads["code"], grads["reconstruction"])
        report[name] = {pair: monitor.record(name, pair, gradient_pair_metrics(a, b))
                        for pair, (a, b) in pairs.items()}
    return {
        "measurement": "weighted_pre_clip_pre_adam",
        "encoder_decoded_scale": encoder_scale,
        "code_routing": "generator_only",
        "window_size": monitor.window_size,
        "window_scope": "current_process_and_phase_sampled_probes",
        "phase": monitor.phase,
        "groups": report,
    }, norms
