import os
import yaml
import autrainer.cli
import yaml

base_dir = "conf"
os.makedirs(base_dir, exist_ok=True)

for seed in [0]:
    exp_name = f"baseline_speech"
    cfg_path = os.path.join(base_dir, f"{exp_name}.yaml")
    with open(f'conf/config.yaml') as f:
            base_cfg = yaml.load(f, Loader=yaml.FullLoader)

    base_cfg['test_augmentation'] = {'group':[]}
    snr_list = []
    for snr in range(0, 15, 5):
        for type_id in range(9):
            snr_list.append(snr)
            base_cfg['test_augmentation']['group'].append({
                    "_target_": "autrainer.augmentations.augmentation_pipeline.AugmentationPipeline",
                    "id": f"111AudioSet{type_id}-{snr}dB",
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

    base_cfg['test_augmentation']['id'] = f'AudioSetBaseline{snr_list[0]}-{snr_list[-1]}'
    base_cfg['hydra']['sweeper']['params']['+seed'] = seed
    base_cfg['hydra']['sweeper']['params']['dataset'] = 'SpeechCommands-16k'
    base_cfg['experiment_id'] = 'speech_baseline'
    with open(cfg_path, "w") as f:
        yaml.dump(base_cfg, f, sort_keys=False)

    print(f"✅ Generated config: {cfg_path}")

    # === 执行训练 ===
    print(f"\n>>> Running experiment: {exp_name} (SNR={snr})\n")
    autrainer.cli.train(
        config_name=f"{exp_name}",
    )