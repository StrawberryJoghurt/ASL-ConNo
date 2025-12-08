import os
import yaml
import autrainer.cli
import yaml

base_dir = "conf"
os.makedirs(base_dir, exist_ok=True)

for seed in [0]:
    # for snr in range(-50, 80, 5):
        # snr=80
    exp_name = f"baseline"
    # exp_name = f"baseline_{snr}"
    cfg_path = os.path.join(base_dir, f"{exp_name}.yaml")
    with open(f'conf/config.yaml') as f:
        base_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    
    # base_cfg["train_augmentation"]={
    #         "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
    #         "id": aug_id,
    #         "pipeline": [
    #             {
    #                 "GaussianNoise":{
    #                 "_target_": "autrainer.augmentations.spectrogram_augmentations.SNR_noise",
    #                 "snr": snr,
    #                 "p": 1.0,
    #                 "generator_seed": 0
    #                 }
    #             }
    #         ],
        # }
    base_cfg['test_augmentation'] = {'group':[]}
    snr_list = []
    for snr in range(-10, 10, 5):
        snr_list.append(snr)
        base_cfg['test_augmentation']['group'].append({
                "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
                "id": f"test({snr})",
                "pipeline": [
                    {
                        "GaussianNoise":{
                        "_target_": "autrainer.augmentations.spectrogram_augmentations.SNR_noise",
                        "snr": snr,
                        "p": 1.0,
                        "generator_seed": 0,
                        "noise_type": "StaticGaussian",
                        }
                    }
                ],        
        })
    base_cfg['test_augmentation']['id'] = f'{snr_list[0]}-{snr_list[-1]}'
    base_cfg['hydra']['sweeper']['params']['+seed'] = seed
    with open(cfg_path, "w") as f:
        yaml.dump(base_cfg, f, sort_keys=False)

    print(f"✅ Generated config: {cfg_path}")

    # === 执行训练 ===
    print(f"\n>>> Running experiment: {exp_name} (SNR={snr})\n")
    autrainer.cli.train(
        config_name=f"{exp_name}",
    )
    # break
