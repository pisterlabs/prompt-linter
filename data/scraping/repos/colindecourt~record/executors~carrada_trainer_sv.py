import imp
import os
import numpy as np

import pytorch_lightning as pl
from torch import nn
import torch

from utils.carrada_functions import get_class_weights, normalize, get_transformations, get_metrics
from utils.loss import CoherenceLoss, SoftDiceLoss
from evaluation.carrada_metrics import Evaluator

class CarradaExecutorSV(pl.LightningModule):
    def __init__(self, config, model, view='range_doppler'):
        """
        PyTorch lightning base class for training SV-RECORD on CARRADA dataset.
        @param config: dictionary with training configuration (lr, optimizer, path to data etc.)
        @param model: instance of the model to train
        @param view: the view to use (range_angle or range_doppler)
        """
        super(CarradaExecutorSV, self).__init__()
        self.scheduler = None
        self.optimizer = None
        self.model = model
        self.view = view

        self.model_cfg = config['model_cfg']
        self.dataset_cfg = config['dataset_cfg']
        self.train_cfg = config['train_cfg']

        weight_path = self.dataset_cfg['weight_path']
        self.project_path = self.dataset_cfg['project_path']

        self.model_name = self.model_cfg['name']
        self.process_signal = self.model_cfg['process_signal']
        self.w_size = self.model_cfg['w_size']
        self.h_size = self.model_cfg['h_size']
        self.nb_classes = self.model_cfg['nb_classes']
        self.in_channels = self.model_cfg['in_channels']

        self.ckpt_dir = self.train_cfg['ckpt_dir']
        self.lr = self.train_cfg['lr']
        self.lr_step = self.train_cfg['lr_step']
        self.loss_step = self.train_cfg['loss_step']
        self.val_step = self.train_cfg['val_step']
        self.viz_step = self.train_cfg['viz_step']
        self.custom_loss = self.train_cfg['loss']
        self.norm_type = self.train_cfg['norm_type']

        # Criterion
        self.criterion = self.define_loss(view, self.custom_loss, weight_path)
        self.nb_losses = len(self.criterion)

        # Metrics
        self.metrics = Evaluator(self.nb_classes)

        self.save_hyperparameters(config)

    def define_loss(self, signal_type, custom_loss, weight_path):
        """
        Method to define the loss to use during training
        @param signal_type: Type of radar view (supported: 'range_doppler', 'range_angle' or 'angle_doppler')
        @param custom_loss: Short name of the custom loss to use (supported: 'wce', 'sdice', 'wce_w10sdice'
                            or 'wce_w10sdice_w5col')
        @param weight_path: path to class weights of dataset
        @return: loss function
        """
        weights = get_class_weights(signal_type, weight_path)
        if custom_loss == 'wce':
            loss = nn.CrossEntropyLoss(weight=weights.float())
        elif custom_loss == 'sdice':
            loss = SoftDiceLoss()
        elif custom_loss == 'wce_w10sdice':
            ce_loss = nn.CrossEntropyLoss(weight=weights.float())
            loss = nn.ModuleList([ce_loss, SoftDiceLoss(global_weight=10.)])
        else:
            loss = nn.CrossEntropyLoss()
        return loss

    def configure_optimizers(self):
        """
        Define optimizer and scheduler
        @return: optimizer and scheduler (if not None)
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9),
            'interval': 'epoch',
            'frequency': 20
        }
        return [self.optimizer], [self.scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_global_prec": 0, "hp/val_global_dice": 0,
                                                   "hp/test_miou": 0, "hp/test_dice": 0,
                                                   "hp/val_loss": 0, "hp/train_loss": 0, })
    def forward(self, x):
        """
        Pytorch Lightning forward pass (inference)
        @param x: input tensor
        @return: segmentation mask
        """
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_id):
        """
        Perform one training step (forward + backward) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        if self.view == 'range_doppler':
            data = batch['rd_matrix'].float()
            mask = batch['rd_mask'].float()
        elif self.view == 'range_angle':
            data = batch['ra_matrix'].float()
            mask = batch['ra_mask'].float()
        data = normalize(data, self.view, norm_type=self.norm_type, proj_path=self.project_path)

        # Forward
        outputs = self.model(data)

        # Case without the CoL
        losses = [c(outputs, torch.argmax(mask, axis=1))
                  for c in self.criterion]
        loss = torch.mean(torch.stack(losses))

        # Log losses
        loss_dict = {
            'train/loss': loss,
            'train/ce': losses[0],
            'train/dice': losses[1],
        }

        self.log_dict(loss_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def on_validation_epoch_end(self):
        test_results = get_metrics(self.metrics)

        if self.view == 'range_doppler':
            test_results_log = {
                'val_metrics/rd_acc': test_results['acc'],
                'val_metrics/rd_prec': test_results['prec'],
                'val_metrics/rd_miou': test_results['miou'],
                'val_metrics/rd_dice': test_results['dice']
            }
        elif self.view == 'range_angle':
            test_results_log = {
                'val_metrics/ra_acc': test_results['acc'],
                'val_metrics/ra_prec': test_results['prec'],
                'val_metrics/ra_miou': test_results['miou'],
                'val_metrics/ra_dice': test_results['dice']
            }

        self.log(test_results_log)
        self.log("hp/val_global_prec", test_results['prec'], on_epoch=True)
        self.log("hp/val_global_dice", test_results['dice'], on_epoch=True)
        self.metrics.reset()

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        if self.view == 'range_doppler':
            data = batch['rd_matrix'].float()
            mask = batch['rd_mask'].float()
        elif self.view == 'range_angle':
            data = batch['ra_matrix'].float()
            mask = batch['ra_mask'].float()
        data = normalize(data, self.view, norm_type=self.norm_type, proj_path=self.project_path)

        outputs = self.forward(data)

        # Compute loss
        # Case without the CoL
        losses = [c(outputs, torch.argmax(mask, axis=1))
                  for c in self.criterion]
        loss = torch.mean(torch.stack(losses))

        # Log losses
        loss_dict = {
            'val/loss': loss,
            'val/ce': losses[0],
            'val/dice': losses[1],
        }

        # Compute metrics
        self.metrics.add_batch(torch.argmax(mask, axis=1).cpu(),
                               torch.argmax(outputs, axis=1).cpu())

        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('hp/val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass + evaluation) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        if self.view == 'range_doppler':
            data = batch['rd_matrix'].float()
            mask = batch['rd_mask'].float()
        elif self.view == 'range_angle':
            data = batch['ra_matrix'].float()
            mask = batch['ra_mask'].float()
        data = normalize(data, self.view, norm_type=self.norm_type, proj_path=self.project_path)

        outputs = self.forward(data)

        # Compute metrics
        self.metrics.add_batch(torch.argmax(mask, axis=1).cpu(),
                               torch.argmax(outputs, axis=1).cpu())

    def on_test_epoch_end(self):
        """
        Compute metrics and log it
        """
        test_results = get_metrics(self.metrics)
        if self.view == 'range_doppler':
            test_results_log = {
                'test_metrics/rd_acc': test_results['acc'],
                'test_metrics/rd_prec': test_results['prec'],
                'test_metrics/rd_miou': test_results['miou'],
                'test_metrics/rd_dice': test_results['dice'],

                'test_metrics/rd_dice_bkg': test_results['dice_by_class'][0],
                'test_metrics/rd_dice_ped': test_results['dice_by_class'][1],
                'test_metrics/rd_dice_cycl': test_results['dice_by_class'][2],
                'test_metrics/rd_dice_car': test_results['dice_by_class'][3],

                'test_metrics/rd_iou_bkg': test_results['miou_by_class'][0],
                'test_metrics/rd_iou_ped': test_results['miou_by_class'][1],
                'test_metrics/rd_iou_cycl': test_results['miou_by_class'][2],
                'test_metrics/rd_iou_car': test_results['miou_by_class'][3],

            }
        elif self.view == 'range_angle':
            test_results_log = {
                'test_metrics/ra_acc': test_results['acc'],
                'test_metrics/ra_prec': test_results['prec'],
                'test_metrics/ra_miou': test_results['miou'],
                'test_metrics/ra_dice': test_results['dice'],

                'test_metrics/ra_dice_bkg': test_results['dice_by_class'][0],
                'test_metrics/ra_dice_ped': test_results['dice_by_class'][1],
                'test_metrics/ra_dice_cycl': test_results['dice_by_class'][2],
                'test_metrics/ra_dice_car': test_results['dice_by_class'][3],

                'test_metrics/ra_iou_bkg': test_results['miou_by_class'][0],
                'test_metrics/ra_iou_ped': test_results['miou_by_class'][1],
                'test_metrics/ra_iou_cycl': test_results['miou_by_class'][2],
                'test_metrics/ra_iou_car': test_results['miou_by_class'][3],
            }

        self.log_dict(test_results_log, on_epoch=True)
        self.log(name='hp/test_dice', value=test_results['dice'], on_epoch=True)
        self.log(name="hp/test_miou", value=test_results['miou'], on_epoch=True)
        self.metrics.reset()
