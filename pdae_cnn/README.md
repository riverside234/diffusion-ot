Either: python -m pdae_cnn.train --config configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml


Other: python -m pdae_cnn.train --domain cat --output-dir outputs/pdae_cnn_cat


Add `--resume` to continue from the latest checkpoint.

The training configs are stored in `configs/stage1a_pdae/`:

- `--domain cat` → `cat_sit_b2_lora_residual_cosmap.yaml`
- `--domain dog` → `dog_sit_b2_lora_residual_cosmap.yaml`

Validate and generate dog reconstruction images on the second GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_pdae_domain.py \
  --domain dog --device cuda:0 --weights ema --no-roundtrip
```
