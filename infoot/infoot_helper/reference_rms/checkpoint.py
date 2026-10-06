import torch

from .tracker import DOMAINS, ReferenceRMSEMA


def load_projection_scales(paths, *, step):
    if set(paths) != set(DOMAINS):
        raise ValueError("Load reference RMS from both cat and dog checkpoints.")
    saved = None
    for name in DOMAINS:
        checkpoint = torch.load(paths[name], map_location="cpu", weights_only=True)
        state = checkpoint.get("projection_rms")
        if state is None:
            raise ValueError(
                f"{paths[name]} has no reference-EMA RMS state; use checkpoints from the updated co-training."
            )
        if (checkpoint.get("step") != step or checkpoint.get("domain") != name
                or int(state["num_updates"]) != step):
            raise ValueError("Reference RMS checkpoint domain or training step does not match.")
        if saved is not None and (state.keys() != saved.keys() or any(
                not torch.equal(state[key], saved[key]) for key in saved)):
            raise ValueError("Cat and dog checkpoints have different reference RMS histories.")
        saved = state
    tracker = ReferenceRMSEMA(decay=float(saved["decay"]), eps=float(saved["eps"]))
    tracker.load_state_dict(saved)
    return tracker.scales()
