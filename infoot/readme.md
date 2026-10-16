Run these commands from the repository root.

Co-training:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 infoot/infoot_cotraining.py
```

Rebuild banks from the new cat/dog PDAE checkpoints, then refit and test:

```bash
python infoot/bank/bank_cat.py --checkpoint outputs/stage1a_cat_rescnn_cosmap/checkpoints/latest.pt
python infoot/bank/bank_dog.py --checkpoint outputs/stage1a_dog_rescnn_cosmap/checkpoints/latest.pt
python infoot/infoot_fit.py --h 0.4 --reg 0.02 --lam 1
python infoot/infoot_test.py --save cnn_bn
```

Test the latest co-training checkpoint:

```bash
python infoot/infoot_test_cotraining.py --save cnn_bn
```

Add `--step 2000` to test a specific co-training checkpoint instead of the latest.
Co-training testing uses the rebuilt banks' latent paths and configuration,
re-encodes them with the selected co-trained models, and restores their saved
matching/mapped BatchNorm statistics from `outputs/infoot_cotraining`.
Use co-training checkpoints with the same encoder architecture as the banks.

These checkpoint paths match the CNN cosmap recipes' `output_dir` values.
Each bank saves a frozen raw-weight
PDAE checkpoint and its configuration under `data/infoot_test`. Encoder BatchNorm
uses its saved running statistics in evaluation mode for both banks and queries.
Refit the plan whenever either bank is rebuilt.
