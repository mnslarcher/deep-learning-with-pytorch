import argparse
import datetime
import os
import sys

from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from util.logconf import logging
from .dsets import LunaDataset
from .model_cls import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into
# metrics_g/metrics_t
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

class LunaTrainingApp():
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--batch-size',
            help="Batch size to use for training",
            default=32,
            type=int
        )
        parser.add_argument(
            '--num-workers',
            help="Number of worker processes for background data loading",
            default=8,
            type=int
        )
        parser.add_argument(
            '--epochs',
            help="Number of epochs to train for",
            default=1,
            type=int
        )
        parser.add_argument(
            '--balanced',
            help="Balance the training data to half benign, half malignant.",
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-flip',
            help=("Augment the training data by randomly flipping the data "
                  "left-right, up-down and front-back."),
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-offset',
            help=("Augment the training data by randomly offsetting the data "
                  "slightly along the X and Y axes."),
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-scale',
            help=("Augment the training data by randomly increasing or "
                  "decreasing the size of the nodule."),
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-rotate',
            help=("Augment the training data by randomly rotating the data "
                  "around the head-foot axis."),
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--augment-noise',
            help=("Augment the training data by randomly adding noise to the "
                  "data."),
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--tb-prefix',
            help=("Data prefix to use for Tensorboard run. Defaults to "
                  "chapter."),
            default='p2ch13'
        )
        parser.add_argument(
            'comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dlwpt'
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA with {} devices."\
                     .format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model = model.to(self.device)

        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced),
            augmentation_dict=self.augmentation_dict
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if
                                                   self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(val_stride=10, isValSet_bool=True)

        val_dl = DataLoader(
            val_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if
                                                   self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            # If samples are loaded in the Dataset on CPU and you would
            # like to push it during training to the GPU, you can speed
            # up the host to device transfer by enabling pin_memory
            pin_memory=self.use_cuda
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix,
                                   self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls-' +
                                            self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-' +
                                            self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1)
            ))


            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            best_score = max(score, best_score)

            self.saveModel('cls', epoch_ndx, score == best_score)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(METRICS_SIZE,
                                   len(train_dl.dataset)).to(self.device)
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Traning".format(epoch_ndx),
            start_ndx=train_dl.num_workers
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx,
                                             batch_tup,
                                             train_dl.batch_size,
                                             trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()
            del loss_var

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE,
                                       len(val_dl.dataset)).to(self.device)
            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation".format(epoch_ndx),
                start_ndx=val_dl.num_workers
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size,
                                      valMetrics_g)

        return valMetrics_g.to('cpu')



    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1]
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        self.initTensorboardWriters()
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))

        metrics_t = metrics_t.detach().numpy()

        benLabel_mask = metrics_t[METRICS_LABEL_NDX] <= 0.5
        benPred_mask = metrics_t[METRICS_PRED_NDX] <= 0.5

        malLabel_mask = ~benLabel_mask
        malPred_mask = ~benPred_mask

        ben_count = benLabel_mask.sum()
        mal_count = malLabel_mask.sum()

        ben_correct = (benLabel_mask & benPred_mask).sum()
        truePos_count = mal_correct = (malLabel_mask & malPred_mask).sum()

        falsePos_count = ben_count - ben_correct
        falseNeg_count = mal_count - mal_correct

        metrics_dict = {}

        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/ben'] = metrics_t[METRICS_LOSS_NDX,
                                             benLabel_mask].mean()
        metrics_dict['loss/mal'] = metrics_t[METRICS_LOSS_NDX,
                                             malLabel_mask].mean()

        metrics_dict['correct/all'] = \
            (mal_correct + ben_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/ben'] = ben_correct / ben_count * 100
        metrics_dict['correct/mal'] = mal_correct / mal_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / (truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = \
            truePos_count / (truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        log.info(("E{} {:8} {loss/all:.4f} loss, "
                  "{correct/all:-5.1f}% correct, "
                  "{pr/precision:.4f} recall, "
                  "{pr/f1_score:.4f} f1 score").format(epoch_ndx,
                                                       mode_str,
                                                       **metrics_dict))
        log.info(("E{} {:8} {loss/ben:.4f} loss, {correct/ben:-5.1f}% "
                  "correct ({ben_correct:} of {ben_count:})") \
                      .format(epoch_ndx,
                              mode_str + '_ben',
                              ben_correct=ben_correct,
                              ben_count=ben_count,
                              **metrics_dict))
        log.info(("E{} {:8} {loss/mal:.4f} loss, {correct/mal:-5.1f}% "
                  "correct ({mal_correct:} of {mal_count:})")\
                        .format(epoch_ndx,
                                mode_str + '_mal',
                                mal_correct=mal_correct,
                                mal_count=mal_count,
                                **metrics_dict))

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count
        )

        bins = [x / 50.0 for x in range(51)]
        benHist_mask = benLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        malHist_mask = malLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if benHist_mask.any():
            writer.add_histogram(
                'is_ben',
                metrics_t[METRICS_PRED_NDX, benHist_mask],
                self.totalTrainingSamples_count,
                bins=bins
            )
        if malHist_mask.any():
            writer.add_histogram(
                'is_mal',
                metrics_t[METRICS_PRED_NDX, malHist_mask],
                self.totalTrainingSamples_count,
                bins=bins
            )

        score = 1 + metrics_t['pr/f1_score'] \
            - metrics_t['loss/mal'] * 0.01 \
            - metrics_t['loss/all'] * 0.0001

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count
            )
        )

        # mode 0o755 => # read/write by me, readable for everone else
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if hasattr(model, 'module'):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            file_path = os.path.join(
                'data-unversioned',
                'part2',
                'models',
                self.cli_args.tb_prefix,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.cli_args.comment,
                    'best'
                )
            )
            torch.save(state, file_path)

            log.debug("Saved model params to {}".format(file_path))


if __name__ == '__main__':
    LunaTrainingApp().main()
