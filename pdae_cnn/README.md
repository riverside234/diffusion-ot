python -m pdae_cnn.train --domain cat --output-dir outputs/pdae_cnn_cat
Add `--resume` to continue from the latest checkpoint.

The training configs are stored in `configs/stage1a_pdae/`:

- `--domain cat` → `cat_sit_b2_lora_residual_cosmap.yaml`
- `--domain dog` → `dog_sit_b2_lora_residual_cosmap.yaml`
