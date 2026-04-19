###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import BreakIdxDataset
from argparse import Namespace
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir
        print(f"saving logs to {self.expdir}")

        root_dir = Path(self.datarc['file_path'])
        meta_data = self.datarc['meta_data']
        max_timestep = self.datarc.get('max_timestep', None)

        self.h5_path = self.datarc.get('h5_path', None)
        glottal_kwargs = self.datarc.get('glottal_kwargs', {"return_glottal": False})
        
        raw_feats_h5_path = self.datarc.get('raw_feats_h5_path', None)
        raw_feats_memmap_dir = self.datarc.get('raw_feats_memmap_dir', None)

        self.train_dataset = BreakIdxDataset('train', root_dir, meta_data, glottal_kwargs, max_timestep, sr=self.datarc.get('sr', 16000))
        self.dev_dataset = BreakIdxDataset('dev', root_dir, meta_data, glottal_kwargs, sr=self.datarc.get('sr', 16000))
        self.test_dataset = BreakIdxDataset('test', root_dir, meta_data, glottal_kwargs, sr=self.datarc.get('sr', 16000))

        model_cls = eval(self.modelrc['select'])
        self.model_cls = model_cls
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.window_size = self.datarc.get('window_size', 0.2) # total window size in seconds
        self.register_buffer('best_score', torch.zeros(1))

        self.feature_rate = self.modelrc.get('feature_rate', 62.5)
        self.labels_mode = self.datarc.get('labels_mode', '0123v4')
        assert all([digit in '01234v' for digit in self.labels_mode]), f"Invalid labels_mode {self.labels_mode}. Supported digits are 0, 1, 2, 3, 4 and 'v' as separator."
        assert all([digit in self.labels_mode for digit in '01234']), f"Labels mode {self.labels_mode} must contain all digits 0, 1, 2, 3, 4."
        self.labels_map = [-1] * 5
        for group_idx, group in enumerate(self.labels_mode.split('v')):
            for digit in group:
                assert digit in '01234', f"Invalid digit {digit} in labels_mode {self.labels_mode}. Supported digits are 0, 1, 2, 3, 4."
                self.labels_map[int(digit)] = group_idx
        self.num_break_idxs = len(set(self.labels_map))
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=self.num_break_idxs,
            **model_conf,
        )

        self.objective = nn.CrossEntropyLoss()

        self.save_metric = self.modelrc.get('save_metric', 'f1')
        assert self.save_metric in ['macro_f1', 'acc', 'f1'], f"Unsupported save_metric {self.save_metric}. Supported metrics are 'macro_f1', 'acc', and 'f1'."

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn, sampler=sampler
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()
    
    def convert_to_boundary_level(self, features, times):
        if "UtteranceLevelBeforeAfter" in str(self.model_cls):
            feats_list = [[], []]
            for fs, ts in zip(features, times):
                for t in ts:
                    st = t - self.window_size / 2
                    et = t + self.window_size / 2
                    st_idx = max(0, int(st * self.feature_rate))
                    midpt_idx = int(t * self.feature_rate)
                    et_idx = min(len(fs), int(et * self.feature_rate))
                    feats_list[0].append(fs[st_idx:midpt_idx])  # before
                    feats_list[1].append(fs[midpt_idx:et_idx])  # after
            boundary_features_len = [torch.IntTensor([len(feat) for feat in feats]).to(device=features[0].device) for feats in feats_list]
            # convert 0s in boundary_features_len to 1s to avoid issues in pooling (feature vectors will be zeros from padding)
            for i in range(2):
                boundary_features_len[i] = torch.where(boundary_features_len[i] == 0, torch.ones_like(boundary_features_len[i]), boundary_features_len[i])
            boundary_features = [pad_sequence(feats, batch_first=True) for feats in feats_list]
            boundary_features = [self.projector(feat) for feat in boundary_features]
        else:
            feats_list = []
            for fs, ts in zip(features, times):
                for t in ts:
                    st = t - self.window_size / 2
                    et = t + self.window_size / 2
                    st_idx = max(0, int(st * self.feature_rate))
                    et_idx = min(len(fs), int(et * self.feature_rate))
                    feats_list.append(fs[st_idx:et_idx])
            boundary_features_len = torch.IntTensor([len(feat) for feat in feats_list]).to(device=features[0].device)
            # convert 0s in boundary_features_len to 1s to avoid issues in pooling (feature vectors will be zeros from padding)
            boundary_features_len = torch.where(boundary_features_len == 0, torch.ones_like(boundary_features_len), boundary_features_len)
            boundary_features = pad_sequence(feats_list, batch_first=True)
            boundary_features = self.projector(boundary_features)
        return boundary_features, boundary_features_len
    
    def forward(self, mode, features, times, labels, filenames, *others, records=None, **kwargs):
        device = features[0].device
        boundary_features, boundary_features_len = self.convert_to_boundary_level(features, times)
        predicted, _ = self.model(boundary_features, boundary_features_len)

        # combine all sub-lists of labels into a single list
        labels = torch.cat([torch.tensor(label, device=device) for label in labels])
        # map labels to the correct format
        labels = torch.LongTensor([self.labels_map[label.item()] for label in labels]).to(device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records['filename'] += filenames
        records['predict_break'] += predicted_classid.cpu().tolist()
        records['truth_break'] += labels.cpu().tolist()

        return loss
    
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'bu_radio_breaks/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == "acc":
                    print(f"\n{mode} {key}: {average}")
                    f.write(f"{mode} at step {global_step}: {average}\n")
                    if mode == "dev" and average > self.best_score and self.save_metric == "acc":
                        self.best_score = torch.ones(1) * average
                        f.write(f"New best on {mode} at step {global_step}: {average}\n")
                        save_names.append(f"{mode}-best.ckpt")

        # Compute and log macro F1 score
        if len(records["predict_break"]) > 0 and len(records["truth_break"]) > 0:
            y_pred = np.array(records["predict_break"])
            y_true = np.array(records["truth_break"])
            if self.num_break_idxs == 2:
                # compute f1 score on class 1
                f1 = f1_score(y_true, y_pred, pos_label=1)
                f1_metric_str = "binary"
            else:
                f1 = f1_score(y_true, y_pred, average="macro")
                f1_metric_str = "macro"
            cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_break_idxs)))
            logger.add_scalar(
                f'bu_radio_breaks/{mode}-{f1_metric_str}_f1',
                f1,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"\n{mode} confusion matrix:\n{cm}\n")
                print(f"\n{mode} {f1_metric_str} F1: {f1}\n")
                f.write(f"\n{mode} confusion matrix:\n{cm}\n")
                f.write(f"{mode} {f1_metric_str} F1 at step {global_step}: {f1}\n")
                if mode == "dev" and f1 > self.best_score and "f1" in self.save_metric:
                    self.best_score = torch.ones(1) * f1
                    f.write(f"New best on {mode} at step {global_step}: {f1}\n")
                    save_names.append(f"{mode}-best.ckpt")

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_break"])]
                file.writelines(lines)
            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {t}\n" for f, t in zip(records["filename"], records["truth_break"])]
                file.writelines(lines)

        return save_names
