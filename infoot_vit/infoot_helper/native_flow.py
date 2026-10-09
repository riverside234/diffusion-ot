from diffusion_ot.training.train_joint_infoot import _reconstruction_loss


def native_flow_loss(domain, x0, v, semantic_dropout=True):
    return _reconstruction_loss(
        domain,
        x0,
        v,
        semantic_dropout=semantic_dropout,
    )