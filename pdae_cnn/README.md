Either: python -m pdae_cnn.train --config configs/stage1a_pdae/cat_sit_b2_lora_residual_cosmap.yaml


Other: python -m pdae_cnn.train --domain cat --output-dir outputs/pdae_cnn_cat


Add `--resume` to continue from the latest checkpoint.

The training configs are stored in `configs/stage1a_pdae/`:

- `--domain cat` → `cat_sit_b2_lora_residual_cosmap.yaml`
- `--domain dog` → `dog_sit_b2_lora_residual_cosmap.yaml`

Both configs enable `augmentation.enabled: true`: source RGB images receive mild
flips, color jitter, and affine transforms, then the frozen VAE re-encodes them.
The CNN and flow target use the same new latent. This requires the original image
dataset and its source metadata, and adds VAE encoding work during training.
Validation uses cached, unaugmented latents. Set `augmentation.enabled: false` to
return to cached-latent training and its configured latent flips.

Validate and generate dog reconstruction images on the second GPU using
`eval/validation.py` and `eval/image.py`:

```bash
CUDA_VISIBLE_DEVICES=1 python -m pdae_cnn.eval \
  --domain dog --device cuda:0 --weights ema
```

This loads the run's `checkpoints/latest.pt`, prints validation metrics, and saves
`results/step_XXXXXX/reconstruction_grid.png` under the run directory.
The grid has eight images per row: original RGB, CFG 1.0, then CFG 1.5.
Both generated rows use the same starting noise and sample order.
Use `--output-dir` for a custom training run, `--checkpoint` for a specific checkpoint,
or `--weights raw` to evaluate without EMA.
