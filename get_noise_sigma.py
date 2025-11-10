
import numpy as np
import os 

DATASET_ROOT = "autrainer-configurations/data"
DATASET_PT = {
    "TIMIT-sentence_type-16k": os.path.join(DATASET_ROOT, "TIMIT-sentence_type/log_mel_16k"),
    "TIMIT-dialect-16k": os.path.join(DATASET_ROOT, "TIMIT-dialect/log_mel_16k"),
    "TIMIT-gender-16k": os.path.join(DATASET_ROOT, "TIMIT-gender/log_mel_16k"),
}

def load_dataset(name):
    path = DATASET_PT[name]
    if name in ['TIMIT-sentence_type-16k', 'TIMIT-dialect-16k', "TIMIT-gender-16k"]:
        data = {
            'TRAIN': {},
            "TEST": {}
        }
        for type in ['TRAIN', 'TEST']:
            sub_root = os.path.join(path, type)
            for dialect in os.listdir(sub_root):
                dialect_root = os.path.join(sub_root, dialect)
                for p_name in os.listdir(dialect_root):
                    p_data = []
                    audi_path = os.path.join(dialect_root, p_name)
                    for audi_name in os.listdir(audi_path):
                        p_data.append(np.load(os.path.join(audi_path, audi_name)))
                    data.setdefault(type, {}).setdefault(dialect, {}).setdefault(p_name, p_data)
        return data
    else:
        raise ValueError("Not implemented") 


def get_gaussian_sigma(p_signal, snr_db):
    p_noise = p_signal / (10 ** (snr_db/10))
    return np.sqrt(p_noise)