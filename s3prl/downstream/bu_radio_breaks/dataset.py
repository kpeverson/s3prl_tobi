from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchaudio
import tqdm

from ...dataset.glottal_extraction import GlottalExtractor

EXCLUDE_IDS = ["data/f3a/labnews/j/radio/f3ajrlp2", "data/f2b/radio/s18/f2bs18p8"]

# BU Radio Corpus break indices classification dataset
class BreakIdxDataset(Dataset):
    def __init__(self, mode, corpus_dir, meta_data, glottal_kwargs, max_timestep=None, sr=16000, **kwargs):
        self.root = corpus_dir
        self.meta_data = meta_data
        self.split_list = open(self.meta_data, "r").readlines()
        self.max_timestep = max_timestep
        self.sr = sr

        self.return_glottal = glottal_kwargs.get("return_glottal", False)
        if self.return_glottal:
            self.glottal_extractor = GlottalExtractor(
                sr=self.sr,
                lpc_window_size=glottal_kwargs.get('lpc_window_size', 0.025),
                lpc_window_stride=glottal_kwargs.get('lpc_window_stride', 0.010),
                lpc_order=glottal_kwargs.get('lpc_order', 16),
                lpc_window=glottal_kwargs.get('lpc_window', 'hamming'),
                lpf_cutoff=glottal_kwargs.get('lpf_cutoff', 1000),
                lpf_order=glottal_kwargs.get('lpf_order', 4),
                half_band_signal=glottal_kwargs.get('half_band_signal', False)
            )

        self.dataset = eval("self.{}".format(mode))()
        self.times_labels = self.get_times_labels(self.dataset)
        all_labels = {i: 0 for i in range(5)}  # Initialize all labels to 0
        for tls in self.times_labels:
            for t, l in tls:
                all_labels[l] += 1
        print(f"[BreakIdxDataset] - labels distribution: {all_labels}")

    def train(self):
        dataset = []
        print(f"[BreakIdxDataset] - Loading training data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 1:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and os.path.exists(os.path.join(self.root, f"{x}.brk")):
                    if x in EXCLUDE_IDS:
                        # print(f"[BreakIdxDataset] - Excluding {x} from training set")
                        continue
                    dataset.append(x)
        print(f"[BreakIdxDataset] - {len(dataset)} training files found")
        return dataset
    
    def dev(self):
        dataset = []
        print(f"[BreakIdxDataset] - Loading development data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 2:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and os.path.exists(os.path.join(self.root, f"{x}.brk")):
                    if x in EXCLUDE_IDS:
                        # print(f"[BreakIdxDataset] - Excluding {x} from dev set")
                        continue
                    dataset.append(x)
        print(f"[BreakIdxDataset] - {len(dataset)} development files found")
        return dataset
    
    def test(self):
        dataset = []
        print(f"[BreakIdxDataset] - Loading test data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 3:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and os.path.exists(os.path.join(self.root, f"{x}.brk")):
                    if x in EXCLUDE_IDS:
                        # print(f"[BreakIdxDataset] - Excluding {x} from test set")
                        continue
                    dataset.append(x)
        print(f"[BreakIdxDataset] - {len(dataset)} test files found")
        return dataset
    
    def get_times_labels(self, dataset):
        all_times_labels = []
        for path in dataset:
            times_labels = []
            brk_path = os.path.join(self.root, f"{path}.brk")
            with open(brk_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            # filter only lines with 3 columns
            lines = [line for line in lines if len(line.split()) == 3]
            for line in lines:
                boundary_time, _, break_idx = line.split()
                boundary_time = float(boundary_time)
                # assert break_idx in [1, 2, 3, 4]
                if break_idx[0] in ["0", "1", "2", "3", "4"]:
                    break_idx = int(break_idx[0])
                    times_labels.append((boundary_time, break_idx))
            all_times_labels.append(times_labels)
        num_breaks = [len(times_labels) for times_labels in all_times_labels]
        print(f"[BreakIdxDataset] - {len(all_times_labels)} files, number of breaks per file, min: {min(num_breaks)}, max: {max(num_breaks)}, avg: {np.mean(num_breaks)}")
        return all_times_labels

    def __len__(self):
        return len(self.dataset)

    def load_audio(self, idx):
        sph_path = os.path.join(self.root, f"{self.dataset[idx]}.sph")
        wav, sr = torchaudio.load(sph_path)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(wav)
            sr = self.sr
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=False)
        wav = wav.squeeze(0)
        if self.return_glottal:
            wav = self.glottal_extractor.extract(wav, sr)
        else:
            wav = wav.numpy()

        return wav

    def __getitem__(self, idx):
        utt_id = os.path.basename(self.dataset[idx])

        wav = self.load_audio(idx)
        length = wav.shape[0]
        
        if self.max_timestep is not None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start:start + self.max_timestep]
                length = self.max_timestep
        
        times_labels = self.times_labels[idx]
        times = [t[0] for t in times_labels]
        labels = [t[1] for t in times_labels]
        if len(times)==0:
            print(f"[BreakIdxDataset] - No breaks found for {self.dataset[idx]}")

        return wav, times, labels, utt_id
    
    def collate_fn(self, samples):
        return zip(*samples)