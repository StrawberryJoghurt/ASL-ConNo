import torchaudio
import torch
import datasets
import io
import numpy as np
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("agkphysics/AudioSet", "balanced")
# Mel 变换器
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=64,
)

amp_to_db = torchaudio.transforms.AmplitudeToDB()

# 统一 mel 时间长度
MAX_FRAMES = 1024   # 可调整

def pad_or_trim(feature, max_len=MAX_FRAMES):
    """
    输入: feature shape (64, T)
    输出: 固定 (64, max_len)
    """
    T = feature.shape[1]

    if T == max_len:
        return feature

    if T > max_len:
        return feature[:, :max_len]

    # pad
    pad_width = max_len - T
    pad = np.zeros((feature.shape[0], pad_width), dtype=feature.dtype)
    return np.concatenate([feature, pad], axis=1)


def preprocess(batch):
    audio = batch["audio"]

    # audio 始终包含 bytes
    waveform, sr = torchaudio.load(io.BytesIO(audio["bytes"]))   # (C, T)

    # 转 mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # mel
    mel = mel_transform(waveform)    # (1, 64, T)
    logmel = amp_to_db(mel)[0].numpy()  # (64, T)

    # 固定长度
    logmel = pad_or_trim(logmel, MAX_FRAMES)

    # 返回
    batch["log_mel"] = logmel  # shape (64, MAX_FRAMES)
    return batch


# 强制 audio 用 bytes
ds = ds.cast_column("audio", datasets.Audio(decode=False))

# 运行
ds_logmel = ds.map(preprocess)

from tqdm import tqdm 
for split in ['train', 'test']:
    for i in tqdm(range(len(ds_logmel[split]))):
        ds_logmel[split][i]['log_mel'] = torch.tensor(ds_logmel[split][i]['log_mel'])

from collections import defaultdict

inverted_index = defaultdict(list)

train_ds = ds_logmel['train']

for i in range(len(train_ds)):
    labels = train_ds[i]['human_labels']   # 可能是 list[str]
    for lb in labels:
        inverted_index[lb].append(i)

from collections import defaultdict
import itertools

train_ds = ds_logmel['train']

# 1. 收集所有标签
all_labels = set()
for i in range(len(train_ds)):
    for lb in train_ds[i]["human_labels"]:
        all_labels.add(lb)
all_labels = sorted(all_labels)

# 2. 初始化矩阵
overlap = {a: defaultdict(int) for a in all_labels}

# 3. 统计重叠
for i in range(len(train_ds)):
    labels = train_ds[i]["human_labels"]
    # 每条样本里的标签两两组合
    for a, b in itertools.combinations(sorted(labels), 2):
        overlap[a][b] += 1
        overlap[b][a] += 1  # 对称矩阵

'Wild animals' in overlap.keys()
'Domestic animals, pets' in overlap.keys()
'Domestic animals, pets' in overlap.keys()

mapping = {
    0:'Wild animals',
    1:'Domestic animals, pets',
    2:'Bell',
    3:'Alarm',
    4:'Wind',
    5:'Water',
    6:'Wild animals+Domestic animals, pets',
    7:'Bell+Alarm',
    8:'Wind+Water',
    # 9:'Pink'
}

from tqdm import tqdm 
audio_index = {}
for split in ['train', 'test']:
    audio_index[split] = {}
    for k in mapping.keys():
        audio_index[split][k] = []

for split in ['train', 'test']:
    print(split)
    for i, sample in enumerate(tqdm(ds_logmel[split])):
        if mapping[0] in sample['human_labels'] or mapping[1] in sample['human_labels']:
            audio_index[split][6].append(i)
            if mapping[0] not in sample['human_labels']:
                audio_index[split][1].append(i)
            if mapping[1] not in sample['human_labels']:
                audio_index[split][0].append(i)
        
        if mapping[2] in sample['human_labels'] or mapping[3] in sample['human_labels']:
            audio_index[split][7].append(i)
            if mapping[2] not in sample['human_labels']:
                audio_index[split][3].append(i)
            if mapping[3] not in sample['human_labels']:
                audio_index[split][2].append(i)
        
        if mapping[4] in sample['human_labels'] or mapping[5] in sample['human_labels']:
            audio_index[split][8].append(i)
            if mapping[4] not in sample['human_labels']:
                audio_index[split][5].append(i)
            if mapping[5] not in sample['human_labels']:
                audio_index[split][4].append(i)


# Structure of AudioSet
print('2')
torch.save({
    'mapping': mapping,
    'audio_index': audio_index,
    'ds_logmel': ds_logmel,
    'overlap': overlap
}, './data/AudioSet/AudioSet.pt')