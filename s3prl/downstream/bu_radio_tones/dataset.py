from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchaudio
import tqdm

from ...dataset.glottal_extraction import GlottalExtractor

# CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')
EXCLUDE_IDS = ["data/f3a/labnews/j/radio/f3ajrlp2", "data/f2b/radio/s18/f2bs18p8"]
ALA_FRAME_SIZE = 0.01 # 10ms frame size for .ala files

# BU Radio Corpus break indices classification dataset
class TonesDataset(Dataset):
    def __init__(self, mode, corpus_dir, meta_data, glottal_kwargs, max_timestep=None, sr=16000, **kwargs):
        self.root = corpus_dir
        self.meta_data = meta_data
        self.split_list = open(self.meta_data, "r").readlines()
        self.max_timestep = max_timestep
        self.sr = sr

        self.dataset = eval("self.{}".format(mode))()
        self.times_labels = self.get_times_labels(self.dataset)
        all_labels = {i: 0 for i in range(2)}  # Initialize all labels to 0
        for tls in self.times_labels:
            for _, _, l in tls:
                all_labels[l] += 1
        print(f"[TonesDataset] - labels distribution: {all_labels}")

        self.return_glottal = glottal_kwargs.get("return_glottal", False)
        if self.return_glottal:
            self.glottal_extractor = GlottalExtractor(
                sr=sr,
                lpc_window_size=glottal_kwargs.get('lpc_window_size', 0.025),
                lpc_window_stride=glottal_kwargs.get('lpc_window_stride', 0.010),
                lpc_order=glottal_kwargs.get('lpc_order', 16),
                lpc_window=glottal_kwargs.get('lpc_window', 'hamming'),
                lpf_cutoff=glottal_kwargs.get('lpf_cutoff', 1000),
                lpf_order=glottal_kwargs.get('lpf_order', 4),
                half_band_signal=glottal_kwargs.get('half_band_signal', False)
            )

    def train(self):
        dataset = []
        print(f"[TonesDataset] - Loading training data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 1:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ton")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ala")):
                    if x in EXCLUDE_IDS:
                        # print(f"[TonesDataset] - Excluding {x} from training set")
                        continue
                    dataset.append(x)
        print(f"[TonesDataset] - {len(dataset)} training files found")
        return dataset
    
    def dev(self):
        dataset = []
        print(f"[TonesDataset] - Loading development data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 2:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ton")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ala")):
                    if x in EXCLUDE_IDS:
                        # print(f"[TonesDataset] - Excluding {x} from dev set")
                        continue
                    dataset.append(x)
        print(f"[TonesDataset] - {len(dataset)} development files found")
        return dataset
    
    def test(self):
        dataset = []
        print(f"[TonesDataset] - Loading test data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 3:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ton")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ala")):
                    if x in EXCLUDE_IDS:
                        # print(f"[TonesDataset] - Excluding {x} from test set")
                        continue
                    dataset.append(x)
        print(f"[TonesDataset] - {len(dataset)} test files found")
        return dataset
    
    def get_times_labels(self, dataset):
        """
        gets syllable start/end times and prominence labels from .ala and .ton files

        syllable start times are determined from vowel phonemes:
            - midpoint of the previous phoneme (if previous phoneme is a consonant)
            - start of the vowel (if previous phoneme is a vowel)
            - start of the file (if vowel is the first phoneme)
        end times are determined similarly
        """
        all_times_labels = []
        for path in dataset:
            # load tones and their timestamps
            times_tones = []
            ton_path = os.path.join(self.root, f"{path}.ton")
            ala_path = os.path.join(self.root, f"{path}.ala")
            with open(ton_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            # filter only lines with 3 columns
            lines = [line for line in lines if len(line.split()) == 3]
            for line in lines:
                time, _, ton = line.split()
                time = float(time)
                times_tones.append((time, ton))
            # load phoneme alignment
            times_labels = []
            with open(ala_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            # filter only lines with 3 columns
            lines = [line.split() for line in lines if len(line.split()) == 3]
            lines = [(line[0], int(line[1]), int(line[1])+int(line[2])) for line in lines] # (phoneme, start_idx, end_idx)
            vowel_idxs = [i for i, line in enumerate(lines) if any([vowel in line[0] for vowel in 'AEIOU'])]
            for i, vidx in enumerate(vowel_idxs):
                # st idx
                if vidx == 0:
                    # vowel is first phoneme - take start of file as st idx
                    st_idx = 0.0
                elif i!=0 and vowel_idxs[i-1] == vidx-1:
                    # previous phone is a vowel - take start of current vowel as st idx
                    st_idx = lines[vidx][1]
                else:
                    # standard case - take center of previous phoneme as st idx
                    st_idx = (lines[vidx-1][1] + lines[vidx-1][2]) / 2.0

                # et idx
                if vidx == len(lines) - 1:
                    # vowel is last phoneme - take end of file as et idx
                    et_idx = lines[-1][2]
                elif i!=len(vowel_idxs)-1 and vowel_idxs[i+1] == vidx+1:
                    # next phone is a vowel - take end of current vowel as et idx
                    et_idx = lines[vidx][2]
                else:
                    # standard case - take center of next phoneme as et idx
                    et_idx = (lines[vidx+1][1] + lines[vidx+1][2]) / 2.0
                st = st_idx * ALA_FRAME_SIZE
                et = et_idx * ALA_FRAME_SIZE
                tones_in_phone = [tt[1] for tt in times_tones if st <= tt[0] <= et]
                if any(["*" in ton for ton in tones_in_phone]):
                    times_labels.append((st, et, 1))
                else:
                    times_labels.append((st, et, 0))

            all_times_labels.append(times_labels)
        num_breaks = [len(times_labels) for times_labels in all_times_labels]
        print(f"[TonesDataset] - {len(all_times_labels)} files, number of breaks per file, min: {min(num_breaks)}, max: {max(num_breaks)}, avg: {np.mean(num_breaks)}")
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
            wav = self.glottal_extractor.extract(wav, idx)
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
        sts = [t[0] for t in times_labels]
        ets = [t[1] for t in times_labels]
        labels = [t[2] for t in times_labels]
        if len(sts)==0:
            print(f"[TonesDataset] - No breaks found for {self.dataset[idx]}")

        return wav, sts, ets, labels, utt_id
    
    def collate_fn(self, samples):
        return zip(*samples)