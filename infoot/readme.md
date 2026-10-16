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

Co-training restores their saved
matching/mapped BatchNorm statistics from `outputs/infoot_cotraining`.

PDAE checkpoint and its configuration under `data/infoot_test`.  
