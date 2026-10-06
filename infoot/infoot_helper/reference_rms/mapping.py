import math

from diffusion_ot.losses.infoot import conditional_density_ratio

from ..infoot import projection


def rms_conditional_mapping(v_query, v_source, v_target, P, h, scales):
    source_scale, target_scale = map(float, scales)
    if any(not math.isfinite(x) or x <= 0 for x in (h, source_scale, target_scale)):
        raise ValueError("Projection bandwidth and RMS scales must be finite and positive.")
    scores = conditional_density_ratio(
        v_query, v_target, v_source, v_target, P.detach(),
        bandwidth=h, distance_scale_x=source_scale, distance_scale_y=target_scale,
    )
    return projection(scores, v_target)
