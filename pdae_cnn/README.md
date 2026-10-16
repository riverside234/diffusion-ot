Either: python -m pdae_cnn.train --config configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml


Or: python -m pdae_cnn.train --domain cat --output-dir outputs/pdae_cnn_cat


Add `--resume` to continue from the latest checkpoint.

The training configs are stored in `configs/stage1a_pdae/`:

- `--domain cat` → `cat_sit_b2_lora_residual_cosmap.yaml`
- `--domain dog` → `dog_sit_b2_lora_residual_cosmap.yaml`


`eval/validation.py` and `eval/image.py`:

```bash
CUDA_VISIBLE_DEVICES=1 python -m pdae_cnn.eval \
  --domain dog --device cuda:0 --weights ema
```

loads the run's `checkpoints/latest.pt`, prints validation metrics, and saves
`results/step_XXXXXX/reconstruction_grid.png` under the run directory.
original RGB, null token (CFG 0.0), CFG 1.0, then CFG 1.5. same starting noise and sample order.
 
