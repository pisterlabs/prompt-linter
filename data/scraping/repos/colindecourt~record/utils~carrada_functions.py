"""
From https://github.com/valeoai/MVRSS
"""
import json
import numpy as np
import torch
import torch.nn as nn
import os
from datasets.carrada.dataloaders import Rescale, Flip, HFlip, VFlip
from utils.loss import CoherenceLoss, SoftDiceLoss


def get_class_weights(signal_type, weight_path):
    """Load class weights for custom loss
    @param signal_type: 'range_doppler' or 'range_angle'
    @param weight_path: path to class weights
    @return: class weights
    """
    if signal_type == 'range_angle':
        file_name = 'ra_weights.json'
    elif signal_type == 'range_doppler':
        file_name = 'rd_weights.json'
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    file_path = os.path.join(weight_path, file_name)
    with open(file_path, 'r') as fp:
        weights = json.load(fp)
    weights = np.array([weights['background'], weights['pedestrian'],
                        weights['cyclist'], weights['car']])
    weights = torch.from_numpy(weights)
    return weights


def normalize(data, signal_type, proj_path, norm_type='local'):
    """
    Method to normalise the radar views
    @param data: radar view
    @param signal_type: signal to normalise ('range_doppler', 'range_angle' and 'angle_doppler')
    @param proj_path: path to the project to load weights
    @param norm_type: type of normalisation to apply ('local' or 'tvt')
    @return: normalised data
    """
    if norm_type in ('local'):
        min_value = torch.min(data)
        max_value = torch.max(data)
        norm_data = torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))
        return norm_data

    elif signal_type == 'range_doppler':
        if norm_type == 'tvt':
            file_path = os.path.join(proj_path, 'configs', 'carrada', 'weights_config', 'rd_stats_all.json')
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            rd_stats = json.load(fp)
        min_value = torch.tensor(rd_stats['min_val'])
        max_value = torch.tensor(rd_stats['max_val'])

    elif signal_type == 'range_angle':
        if norm_type == 'tvt':
            file_path = os.path.join(proj_path, 'configs', 'carrada', 'weights_config', 'ra_stats_all.json')
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ra_stats = json.load(fp)
        min_value = torch.tensor(ra_stats['min_val'])
        max_value = torch.tensor(ra_stats['max_val'])

    elif signal_type == 'angle_doppler':
        if norm_type == 'tvt':
            file_path = os.path.join(proj_path, 'configs', 'carrada', 'weights_config', 'ad_stats_all.json')
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ad_stats = json.load(fp)
        min_value = torch.tensor(ad_stats['min_val'])
        max_value = torch.tensor(ad_stats['max_val'])

    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))

    norm_data = torch.div(torch.sub(data, min_value),
                          torch.sub(max_value, min_value))
    return norm_data


def get_transformations(transform_names, split='train', sizes=None):
    """Create a list of functions used for preprocessing
    @param transform_names: list of str, one for each transformation
    @param split: split currently used
    @param sizes: int or tuple (optional)
    @return: transformations to use for preprocessing (e.g. data augmentation)
    """
    transformations = list()
    if 'rescale' in transform_names:
        transformations.append(Rescale(sizes))
    if 'flip' in transform_names and split == 'train':
        transformations.append(Flip(0.5))
    if 'vflip' in transform_names and split == 'train':
        transformations.append(VFlip())
    if 'hflip' in transform_names and split == 'train':
        transformations.append(HFlip())
    return transformations


def get_metrics(metrics):
    """Structure the metric results
    @param metrics: contains statistics recorded during inference
    @return: metrics values
    """
    metrics_values = dict()
    acc, acc_by_class = metrics.get_pixel_acc_class()  # harmonic_mean=True)
    prec, prec_by_class = metrics.get_pixel_prec_class()
    recall, recall_by_class = metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = metrics.get_dice_class()
    metrics_values['acc'] = acc
    metrics_values['acc_by_class'] = acc_by_class.tolist()
    metrics_values['prec'] = prec
    metrics_values['prec_by_class'] = prec_by_class.tolist()
    metrics_values['recall'] = recall
    metrics_values['recall_by_class'] = recall_by_class.tolist()
    metrics_values['miou'] = miou
    metrics_values['miou_by_class'] = miou_by_class.tolist()
    metrics_values['dice'] = dice
    metrics_values['dice_by_class'] = dice_by_class.tolist()
    return metrics_values