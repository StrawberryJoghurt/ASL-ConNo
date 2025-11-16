import os
import yaml
import autrainer.cli
import yaml
# === 参数定义 ===
experiments = [
    ("GaussianNoise(neg20)", 5.2243),
    ("GaussianNoise(H)", 0.5224),
    ("GaussianNoise(L)", 0.0294),
]

base_dir = "conf"
os.makedirs(base_dir, exist_ok=True)

for post in ['neg_20', 'H', 'L']:
    for aug_id, std in experiments:
        
        exp_name = f"{aug_id}_{post}"
        cfg_path = os.path.join(base_dir, f"{exp_name}.yaml")
        with open(f'conf/baseline_{post}.yaml') as f:
            base_cfg = yaml.load(f, Loader=yaml.FullLoader)
        # cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        # === 写配置文件 ===
        
        base_cfg["train_augmentation"]={
                "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
                "id": aug_id,
                "pipeline": [
                    {
                        "GaussianNoise":{
                        "_target_": "autrainer.augmentations.spectrogram_augmentations.GaussianNoise",
                        "mean": 0.0,
                        "std": std,
                        "p": 1.0,
                        "generator_seed": 0
                        }
                    }
                ],
        }
        

        with open(cfg_path, "w") as f:
            yaml.dump(base_cfg, f, sort_keys=False)

        print(f"✅ Generated config: {cfg_path}")

        # === 执行训练 ===
        print(f"\n>>> Running experiment: {exp_name} (std={std})\n")
        autrainer.cli.train(
            config_name=f"{exp_name}",
        )
