"""Cache cat/dog train/validation RGB images with the bundled SiT-L VAE."""
from cache_vae_latents import main as cache_main


def main(argv=None) -> int:
    return cache_main(argv, default_data_config="configs/data/afhq_pdae_v2_l.yaml",
                      default_model_config="configs/model/sit_l2_256.yaml")


if __name__ == "__main__":
    raise SystemExit(main())
