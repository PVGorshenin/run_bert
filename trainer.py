import numpy as np
import os
import torch
from .custom_logger import CustomLogger
from tqdm import tqdm


OUTDIR = "../data/result/"


class BertPredictor(object):

    def __init__(self, model, train_loader, criterion, optimizer, split_rand_state, metric, description, device,
                 val_loader=None, epochs_count=1, result_dir="./data/result/", num_labels=1):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.i_epoch = 0
        self.epochs_count = epochs_count
        if val_loader is not None:
            val_loader_num = val_loader.num
        else:
            val_loader_num = 0
        self.logger = CustomLogger(self,
                                   train_loader_num=train_loader.num,
                                   val_loader_num=val_loader_num,
                                   epochs_count=epochs_count,
                                   num_labels=num_labels,
                                   metric=metric,
                                   result_dir=result_dir,
                                   split_rand_state=split_rand_state,
                                   description=description
                                   )

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def fit(self):
        self.logger._make_resultdir_n_subdirs()
        self.logger.save_description()
        self.logger.save_loaders_information()
        for i_epoch in range(self.epochs_count):
            name_prefix = '[{} / {}] '.format(i_epoch + 1, self.epochs_count)
            self.do_epoch(name_prefix + 'Train:')
            if self.val_loader is not None:
                self.predict_n_save_val()
            if not self.i_epoch:
                self.logger.save_labels()
            self.i_epoch += 1
        self.logger.save_metric_scores()

    def do_epoch(self, name=None):
        epoch_loss = 0
        name = name or ''
        self.model.train(True)
        batches_count = len(self.train_loader)
        if self.logger.result_dir == '':
            self.logger._make_resultdir_n_subdirs()
        with torch.autograd.set_grad_enabled(True):
            with tqdm(total=batches_count) as progress_bar:
                for batch in self.train_loader:
                    df_idx, batch_id, batch_att, batch_seg, batch_label = batch.values()
                    logits = self.model(ids=batch_id.to(self.device),
                                        token_type_ids=batch_seg.to(self.device),
                                        attention_mask=batch_att.to(self.device))
                    loss = self.criterion(logits.cpu(), batch_label.cpu().reshape(-1, 1))
                    epoch_loss += loss.item()
                    self.logger.accumulate_train_preds(logits, batch_label, df_idx)

                    if self.optimizer:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    progress_bar.update()
                    progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(name, loss.item()))

                progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(name, epoch_loss / batches_count))
                self.logger.save_train_lr()
                self.logger.save_train_preds()
                self.logger.save_models()
                self.logger.extend_metric_lst_by_epoch(is_train=True)
        return epoch_loss / batches_count

    def predict_n_save_val(self):
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                df_idx, batch_id, batch_att, batch_seg, batch_label = batch.values()
                logits = self.model(ids=batch_id.to(self.device),
                                    token_type_ids=batch_seg.to(self.device),
                                    attention_mask=batch_att.to(self.device))
                self.logger.accumulate_val_preds(logits, batch_label, df_idx)
            self.logger.extend_metric_lst_by_epoch(is_train=False)
            self.logger.save_val_preds()