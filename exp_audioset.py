import os
import yaml
import autrainer.cli
import yaml

base_dir = "conf"
os.makedirs(base_dir, exist_ok=True)
train_test_type = {
     0: [0,1],
     1: [0,1],
     2: [2,3],
     3: [2,3],
     4: [4,5],
     5: [4,5],
     6: [6,7,8],
     7: [6,7,8],
     8: [6,7,8]
}
for snr_train in range(0,20,5):
    for noise_type_train in train_test_type.keys():    
        for seed in [0]:
            exp_name = f"baseline_audioset_train{noise_type_train}({snr_train}dB)"
            cfg_path = os.path.join(base_dir, f"{exp_name}.yaml")
            with open(f'conf/config.yaml') as f:
                    base_cfg = yaml.load(f, Loader=yaml.FullLoader)
            base_cfg['train_augmentation'] = {
                    "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
                    "id": f"AudioSet{noise_type_train}({snr_train})",
                    "pipeline": [
                        {
                            "AudioSet":{
                            "_target_": "autrainer.augmentations.spectrogram_augmentations.SNR_noise",
                            "snr": snr_train,
                            "p": 1.0,
                            "generator_seed": 0,
                            "noise_type": f"AudioSet{noise_type_train}",
                            }
                        }
                    ],
            }
            base_cfg['test_augmentation'] = {'group':[]}
            snr_list = []
            for snr in range(0, 20, 5):
                for type_id in train_test_type[noise_type_train]:
                    snr_list.append(snr)
                    base_cfg['test_augmentation']['group'].append({
                            "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
                            "id": f"AudioSet{type_id}-{snr}dB",
                            "pipeline": [
                                {
                                    "AudioSet":{
                                    "_target_": "autrainer.augmentations.spectrogram_augmentations.SNR_noise",
                                    "snr": snr,
                                    "p": 1.0,
                                    "generator_seed": 0,
                                    "noise_type": f"AudioSet{type_id}",
                                    }
                                }
                            ],        
                    })

            base_cfg['test_augmentation']['id'] = f'AudioSet{snr_list[0]}-{snr_list[-1]}'
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
