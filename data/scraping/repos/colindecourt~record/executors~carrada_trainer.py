import imp
import os
import numpy as np 

import pytorch_lightning as pl
from torch import nn
import torch

from utils.carrada_functions import get_class_weights, normalize, get_transformations, get_metrics
from utils.loss import CoherenceLoss, SoftDiceLoss
from evaluation.carrada_metrics import Evaluator


class CarradaExecutor(pl.LightningModule):
    def __init__(self, config, model):
        """
        PyTorch lightning base class for training MV-RECORD on CARRADA dataset.
        @param config: dictionary with training configuration (lr, optimizer, path to data etc.)
        @param model: instance of the model to train
        """
        super(CarradaExecutor, self).__init__()
        self.scheduler = None
        self.optimizer = None
        self.model = model

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
        self.rd_criterion = self.define_loss('range_doppler', self.custom_loss, weight_path)
        self.ra_criterion = self.define_loss('range_angle', self.custom_loss, weight_path)
        self.nb_losses = len(self.rd_criterion)

        # Metrics
        self.rd_metrics = Evaluator(self.nb_classes)
        self.ra_metrics = Evaluator(self.nb_classes)

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9),
            'interval': 'epoch',
            'frequency': 20
        }        
        return [self.optimizer], [self.scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/val_global_prec": 0, "hp/val_global_dice": 0,
                                                   "hp/test_rd_miou": 0, "hp/test_ra_miou": 0,
                                                   "hp/val_loss": 0, "hp/train_loss": 0, })

    def forward(self, rd_data, ra_data, ad_data=None):
        """
        Pytorch Lightning forward pass (inference)
        @param rd_data: range doppler tensor
        @param ra_data: range angle tensor
        @param ad_data: angle doppler tensor
        @return: range doppler and range angle segmentation masks
        """
        rd_outputs, ra_outputs = self.model(rd_data, ra_data, ad_data)
        return rd_outputs, ra_outputs
    
    def training_step(self, batch, batch_id):
        """
        Perform one training step (forward + backward) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        rd_data = batch['rd_matrix'].float()
        ra_data = batch['ra_matrix'].float()
        ad_data = batch['ad_matrix'].float()
        rd_mask = batch['rd_mask'].float()
        ra_mask = batch['ra_mask'].float()
        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type, proj_path=self.project_path)
        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type, proj_path=self.project_path)

        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type, proj_path=self.project_path)
        
        # Forward
        rd_outputs, ra_outputs = self.model(rd_data, ra_data, ad_data)

        # Compute loss 
        if self.nb_losses < 3:
            # Case without the CoL
            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                            for c in self.rd_criterion]
            rd_loss = torch.mean(torch.stack(rd_losses))
            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                            for c in self.ra_criterion]
            ra_loss = torch.mean(torch.stack(ra_losses))
            loss = torch.mean(rd_loss + ra_loss)
        else:
            # Case with the CoL
            # Select the wCE and wSDice
            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                            for c in self.rd_criterion[:2]]
            rd_loss = torch.mean(torch.stack(rd_losses))
            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                            for c in self.ra_criterion[:2]]
            ra_loss = torch.mean(torch.stack(ra_losses))
            # Coherence loss
            coherence_loss = self.rd_criterion[2](rd_outputs, ra_outputs)
            loss = torch.mean(rd_loss + ra_loss + coherence_loss)
        
        # Log losses
        if self.nb_losses > 2:
            loss_dict = {
                'train/loss': loss,
                'train/rd_global': rd_loss,
                'train/rd_ce': rd_losses[0],
                'train/rd_Dice': rd_losses[1],
                'train/ra_global': ra_loss,
                'train/ra_ce': ra_losses[0],
                'train/ra_Dice': ra_losses[1],
                'train/coherence': coherence_loss
            }
        else:
            loss_dict = {
                'train/loss': loss,
                'train/rd_global': rd_loss,
                'train/rd_ce': rd_losses[0],
                'train/rd_Dice': rd_losses[1],
                'train/ra_global': ra_loss,
                'train/ra_ce': ra_losses[0],
                'train/ra_Dice': ra_losses[1]
            }
        self.log_dict(loss_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_validation_epoch_end(self):
        test_results = dict()
        test_results['range_doppler'] = get_metrics(self.rd_metrics)
        test_results['range_angle'] = get_metrics(self.ra_metrics)

        test_results['global_acc'] = (1/2)*(test_results['range_doppler']['acc']+
                                                    test_results['range_angle']['acc'])
        test_results['global_prec'] = (1/2)*(test_results['range_doppler']['prec']+
                                                    test_results['range_angle']['prec'])
        test_results['global_dice'] = (1/2)*(test_results['range_doppler']['dice']+
                                                    test_results['range_angle']['dice'])

        test_results_log = {
            'val_metrics/rd_acc': test_results['range_doppler']['acc'],
            'val_metrics/rd_prec': test_results['range_doppler']['prec'],
            'val_metrics/rd_miou': test_results['range_doppler']['miou'],
            'val_metrics/rd_dice': test_results['range_doppler']['dice'],
            'val_metrics/ra_acc': test_results['range_angle']['acc'],
            'val_metrics/ra_prec': test_results['range_angle']['prec'],
            'val_metrics/ra_miou': test_results['range_angle']['miou'],
            'val_metrics/ra_dice': test_results['range_angle']['dice'],
            'val_metrics/global_prec': test_results['global_prec'],
            'val_metrics/global_acc': test_results['global_acc'],
            'val_metrics/global_dice': test_results['global_dice']
        }

        self.log_dict(test_results_log, on_epoch=True)
        self.log("hp/val_global_prec", test_results['global_prec'], on_epoch=True)
        self.log("hp/val_global_dice", test_results['global_dice'], on_epoch=True)
        self.rd_metrics.reset()
        self.ra_metrics.reset()

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        rd_data = batch['rd_matrix'].float()
        ra_data = batch['ra_matrix'].float()
        ad_data = batch['ad_matrix'].float()
        rd_mask = batch['rd_mask'].float()
        ra_mask = batch['ra_mask'].float()
        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type, proj_path=self.project_path)
        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type, proj_path=self.project_path)

        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type, proj_path=self.project_path)

        rd_outputs, ra_outputs = self.forward(rd_data, ra_data, ad_data)
            
        # Compute loss 
        if self.nb_losses < 3:
            # Case without the CoL
            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                            for c in self.rd_criterion]
            rd_loss = torch.mean(torch.stack(rd_losses))
            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                            for c in self.ra_criterion]
            ra_loss = torch.mean(torch.stack(ra_losses))
            loss = torch.mean(rd_loss + ra_loss)
        else:
            # Case with the CoL
            # Select the wCE and wSDice
            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                            for c in self.rd_criterion[:2]]
            rd_loss = torch.mean(torch.stack(rd_losses))
            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                            for c in self.ra_criterion[:2]]
            ra_loss = torch.mean(torch.stack(ra_losses))
            # Coherence loss
            coherence_loss = self.rd_criterion[2](rd_outputs, ra_outputs)
            loss = torch.mean(rd_loss + ra_loss + coherence_loss)

        # Compute metrics
        self.rd_metrics.add_batch(torch.argmax(rd_mask, axis=1).cpu(),
                                torch.argmax(rd_outputs, axis=1).cpu())
        self.ra_metrics.add_batch(torch.argmax(ra_mask, axis=1).cpu(),
                                torch.argmax(ra_outputs, axis=1).cpu())
        
        if self.nb_losses > 2:
            loss_dict = {
                'val/loss': loss,
                'val/rd_global': rd_loss,
                'val/rd_ce': rd_losses[0],
                'val/rd_Dice': rd_losses[1],
                'val/ra_global': ra_loss,
                'val/ra_ce': ra_losses[0],
                'val/ra_Dice': ra_losses[1],
                'val/coherence': coherence_loss
            }
        else:
            loss_dict = {
                'val/loss': loss,
                'val/rd_global': rd_loss,
                'val/rd_ce': rd_losses[0],
                'val/rd_Dice': rd_losses[1],
                'val/ra_global': ra_loss,
                'val/ra_ce': ra_losses[0],
                'val/ra_Dice': ra_losses[1]      
            }
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('hp/val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """

        rd_data = batch['rd_matrix'].float()
        ra_data = batch['ra_matrix'].float()
        ad_data = batch['ad_matrix'].float()
        rd_mask = batch['rd_mask'].float()
        ra_mask = batch['ra_mask'].float()
        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type, proj_path=self.project_path)
        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type, proj_path=self.project_path)

        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type, proj_path=self.project_path)

        rd_outputs, ra_outputs = self.forward(rd_data, ra_data, ad_data)

        # Compute metrics
        self.rd_metrics.add_batch(torch.argmax(rd_mask, axis=1).cpu(),
                                  torch.argmax(rd_outputs, axis=1).cpu())
        self.ra_metrics.add_batch(torch.argmax(ra_mask, axis=1).cpu(),
                                  torch.argmax(ra_outputs, axis=1).cpu())

    def on_test_epoch_end(self):
        """
        Compute metrics and log it 
        """
        test_results = dict()
        test_results['range_doppler'] = get_metrics(self.rd_metrics)
        test_results['range_angle'] = get_metrics(self.ra_metrics)

        test_results['global_acc'] = (1/2)*(test_results['range_doppler']['acc']+
                                                    test_results['range_angle']['acc'])
        test_results['global_prec'] = (1/2)*(test_results['range_doppler']['prec']+
                                                    test_results['range_angle']['prec'])
        test_results['global_dice'] = (1/2)*(test_results['range_doppler']['dice']+
                                                    test_results['range_angle']['dice'])

        test_results_log = {
            'test_metrics/rd_acc': test_results['range_doppler']['acc'],
            'test_metrics/rd_prec': test_results['range_doppler']['prec'],
            'test_metrics/rd_miou': test_results['range_doppler']['miou'],
            'test_metrics/rd_dice': test_results['range_doppler']['dice'],
            'test_metrics/ra_acc': test_results['range_angle']['acc'],
            'test_metrics/ra_prec': test_results['range_angle']['prec'],
            'test_metrics/ra_miou': test_results['range_angle']['miou'],
            'test_metrics/ra_dice': test_results['range_angle']['dice'],
            'test_metrics/global_prec': test_results['global_prec'],
            'test_metrics/global_acc': test_results['global_acc'],
            'test_metrics/global_dice': test_results['global_dice'],
            'test_metrics/rd_dice_bkg': test_results['range_doppler']['dice_by_class'][0],
            'test_metrics/rd_dice_ped': test_results['range_doppler']['dice_by_class'][1],
            'test_metrics/rd_dice_cycl': test_results['range_doppler']['dice_by_class'][2],
            'test_metrics/rd_dice_car': test_results['range_doppler']['dice_by_class'][3],
            'test_metrics/ra_dice_bkg': test_results['range_angle']['dice_by_class'][0],
            'test_metrics/ra_dice_ped': test_results['range_angle']['dice_by_class'][1],
            'test_metrics/ra_dice_cycl': test_results['range_angle']['dice_by_class'][2],
            'test_metrics/ra_dice_car': test_results['range_angle']['dice_by_class'][3],
            
            'test_metrics/rd_iou_bkg': test_results['range_doppler']['miou_by_class'][0],
            'test_metrics/rd_iou_ped': test_results['range_doppler']['miou_by_class'][1],
            'test_metrics/rd_iou_cycl': test_results['range_doppler']['miou_by_class'][2],
            'test_metrics/rd_iou_car': test_results['range_doppler']['miou_by_class'][3],
            'test_metrics/ra_iou_bkg': test_results['range_angle']['miou_by_class'][0],
            'test_metrics/ra_iou_ped': test_results['range_angle']['miou_by_class'][1],
            'test_metrics/ra_iou_cycl': test_results['range_angle']['miou_by_class'][2],
            'test_metrics/ra_iou_car': test_results['range_angle']['miou_by_class'][3],
            
        }

        self.log_dict(test_results_log, on_epoch=True)
        self.log(name='hp/test_rd_miou', value=test_results['range_doppler']['miou'], on_epoch=True)
        self.log(name="hp/test_ra_miou", value=test_results['range_angle']['miou'], on_epoch=True)
        self.rd_metrics.reset()
        self.ra_metrics.reset()