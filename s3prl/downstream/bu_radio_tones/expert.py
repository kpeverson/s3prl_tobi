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
from .dataset import TonesDataset
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
        glottal_kwargs = self.datarc.get('glottal_kwargs', {"return_glottal": False})
        max_timestep = self.datarc.get('max_timestep', None)

        self.train_dataset = TonesDataset('train', root_dir, meta_data, glottal_kwargs, max_timestep, sr=self.datarc.get('sr', 16000))
        self.dev_dataset = TonesDataset('dev', root_dir, meta_data, glottal_kwargs, sr=self.datarc.get('sr', 16000))
        self.test_dataset = TonesDataset('test', root_dir, meta_data, glottal_kwargs, sr=self.datarc.get('sr', 16000))

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.window_size = self.datarc.get('window_size', 0.2) # total window size in seconds
        self.register_buffer('best_score', torch.zeros(1))

        self.feature_rate = self.modelrc.get('feature_rate', 62.5)

        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=1,
            **model_conf,
        )

        # use binary cross entropy loss
        self.objective = nn.BCEWithLogitsLoss()

        self.save_metric = 'f1'

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
    
    def convert_to_syl_level(self, features, sts, ets):
        feats_list = []
        for fs, st, et in zip(features, sts, ets):
            for s, e in zip(st, et):
                s_idx = max(0, int(s * self.feature_rate))
                e_idx = min(len(fs), int(e * self.feature_rate))
                feats_list.append(fs[s_idx:e_idx])
        feats_len = torch.IntTensor([len(feat) for feat in feats_list]).to(features[0].device)
        # convert 0s in feats_len to 1s to avoid issues in pooling (feature vectors will be zeros from padding)
        feats_len = torch.where(feats_len == 0, torch.ones_like(feats_len), feats_len)
        feats = pad_sequence(feats_list, batch_first=True)
        feats = self.projector(feats)
        return feats, feats_len
    
    def forward(self, mode, features, sts, ets, labels, filenames, *others, records=None, **kwargs):
        device = features[0].device
        word_features, word_features_len = self.convert_to_syl_level(features, sts, ets)
        predicted, _ = self.model(word_features, word_features_len)

        # combine all sub-lists of labels into a single list
        labels = torch.cat([torch.tensor(label, device=device) for label in labels])
        loss = self.objective(predicted.squeeze(-1), labels.float())

        # get predicted class ids using threshold 0
        predicted_classid = (predicted > 0).long().squeeze(-1)
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records['filename'] += filenames
        records['predict_label'] += predicted_classid.cpu().tolist()
        records['truth_label'] += labels.cpu().tolist()

        return loss
    
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'bu_radio_tones/{mode}-{key}',
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
        if len(records["predict_label"]) > 0 and len(records["truth_label"]) > 0:
            y_pred = np.array(records["predict_label"])
            y_true = np.array(records["truth_label"])
            f1 = f1_score(y_true, y_pred) #, average="macro")
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            logger.add_scalar(
                f'bu_radio_tones/{mode}-f1',
                f1,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"\n{mode} confusion matrix:\n{cm}")
                print(f"\n{mode} F1: {f1}\n")
                f.write(f"{mode} F1 at step {global_step}: {f1}\n")

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_label"])]
                file.writelines(lines)
            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {t}\n" for f, t in zip(records["filename"], records["truth_label"])]
                file.writelines(lines)

        return save_names
