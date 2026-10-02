"""Launch defaults for fresh residual-cosmap work; legacy recipes stay explicit."""
STAGE1A_EVAL = "configs/stage1a_eval/residual_sit_b2_256.yaml"
STAGE1B_TRAIN = "configs/stage1b_infoot/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml"
STAGE1B_EVAL = "configs/stage1b_eval/self_supervised_infonce_v7_residual_cosmap_sit_b2.yaml"


def stage1a_training_config(domain):
    if domain not in {"cat", "dog"}:
        raise ValueError("Choose cat or dog for the default Stage 1A recipe.")
    return f"configs/stage1a_pdae/{domain}_sit_b2_lora_residual_cosmap.yaml"
