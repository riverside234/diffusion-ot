"""Download the pinned SiT-L/2 diffusion model and its bundled VAE using HF Hub.

Reuses the repository snapshot verifier. huggingface_hub reads HF_TOKEN when set.
"""
from verify_sit_snapshot import main as verify_main


def main(argv=None) -> int:
    return verify_main(argv, default_config="configs/pretrained/bilisakura_sit_l2_256.yaml",
                       download_by_default=True)


if __name__ == "__main__":
    raise SystemExit(main())
