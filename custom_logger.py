import numpy as np
import os
import torch
from common_ import to_numpy
from datetime import datetime


class CustomLogger():
    def __init__(self, predictor, train_loader_num, val_loader_num, epochs_count, num_labels, metric, result_dir,
                 split_rand_state, description):

        self.predictor = predictor
        self.train_loader_num = train_loader_num
        self.val_loader_num = val_loader_num
        self.np_train_preds = np.zeros((train_loader_num, num_labels, epochs_count))
        self.np_train_targets = np.zeros((train_loader_num))
        if val_loader_num:
            self.np_val_preds = np.zeros((val_loader_num, num_labels, epochs_count))
            self.np_val_targets = np.zeros((val_loader_num))
        self.metric_score_lst = []
        self.val_metric_score_lst = []
        self.split_rand_state = split_rand_state
        self.result_dir = result_dir
        self.metric = metric
        self.description = description

    def _calc_metric(self, logits, batch_labels):
        return self.metric(batch_labels, logits)

    def save_description(self):
        with open(os.path.join(self.result_dir, 'meta.txt'), 'w') as meta_file:
            meta_file.writelines(self.description + '\n')

    def save_metric_scores(self):
        np.savetxt(os.path.join(self.result_dir, 'val_scores.csv'), self.val_metric_score_lst,
                   delimiter=",")
        np.savetxt(os.path.join(self.result_dir, 'train_scores.csv'), self.metric_score_lst,
                   delimiter=",")

    def _accumulate_preds(self, logits, batch_label, df_idx, preds, targets):
        if self.predictor.i_epoch:
            preds[df_idx, :, self.predictor.i_epoch] = to_numpy(logits)
        else:
            preds[df_idx, :, self.predictor.i_epoch] = to_numpy(logits)
            targets[df_idx] = to_numpy(batch_label)

    def accumulate_train_preds(self, logits, batch_label, df_idx):
        self._accumulate_preds(logits, batch_label, df_idx, self.np_train_preds, self.np_train_targets)

    def accumulate_val_preds(self, logits, batch_label, df_idx):
        self._accumulate_preds(logits, batch_label, df_idx, self.np_val_preds, self.np_val_targets)

    def _make_resultdir_n_subdirs(self):
        """
        Creates result_dir and datetime named folder inside
        """
        now = datetime.now()
        datename = "-".join([str(now.date())[5:], str(now.hour)])
        self.result_dir = os.path.join(self.result_dir, datename)
        print(self.result_dir)
        if not os.path.isdir(self.result_dir):
            os.makedirs(os.path.join(self.result_dir, 'preds'))
            os.makedirs(os.path.join(self.result_dir, 'models'))

    def save_train_lr(self):
        """
        Saves lr, train_preds and model
        """
        with open(os.path.join(self.result_dir, 'meta.txt'), 'a') as meta_file:
            current_lr = self.predictor._get_lr()
            meta_file.writelines(f'\n i_epoch --> {self.predictor.i_epoch}  lr on end --> {current_lr}')

    def save_train_preds(self):
        np.savetxt(os.path.join(self.result_dir, 'preds', f'train_preds_{self.predictor.i_epoch}.csv'),
                   self.np_train_preds[:, :, self.predictor.i_epoch], delimiter=",")

    def save_val_preds(self):
        np.savetxt(os.path.join(self.result_dir, 'preds', f'val_preds_{self.predictor.i_epoch}.csv'),
                   self.np_val_preds[:, :, self.predictor.i_epoch], delimiter=",")

    def save_models(self):
        torch.save(self.predictor.model.state_dict(), os.path.join(self.result_dir, 'models',
                                                                   f'model_epoch{self.predictor.i_epoch}'))

    def save_labels(self):
        np.savetxt(os.path.join(self.result_dir, 'preds', f'val_labels.csv'), self.np_val_targets, delimiter=",")
        np.savetxt(os.path.join(self.result_dir, 'preds', f'train_labels.csv'),self.np_train_targets, delimiter=",")

    def save_loaders_information(self):
        with open(os.path.join(self.result_dir, 'meta.txt'), 'a') as meta_file:
            meta_file.writelines(f'train loader len --> {self.train_loader_num} \n')
            meta_file.write(f'split_rand_state --> {self.split_rand_state}\n')
            if self.val_loader_num:
                meta_file.writelines(f'val loader len --> {self.val_loader_num} \n')

    def extend_metric_lst_by_epoch(self, is_train=True):
        if is_train:
            metric_score = self._calc_metric(self.np_train_preds[:, :, self.predictor.i_epoch], self.np_train_targets)
            self.metric_score_lst.append(metric_score)
        else:
            val_metric_score = self._calc_metric(self.np_val_preds[:, :, self.predictor.i_epoch],  self.np_val_targets)
            print('val_metric_score -->', val_metric_score)
            self.val_metric_score_lst.append(val_metric_score)