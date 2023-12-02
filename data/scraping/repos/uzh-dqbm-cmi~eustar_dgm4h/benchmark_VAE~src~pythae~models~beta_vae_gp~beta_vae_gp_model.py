import os
from typing import Optional

import torch
import torch.nn.functional as F
import math
import cloudpickle
import inspect
import sys
from copy import deepcopy
from ...data.datasets import BaseDataset, MissingDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .beta_vae_gp_config import BetaVAEgpConfig

from ..base.base_config import BaseAEConfig, EnvironmentConfig

from .GP import PriorGP

from .encoder_decoder import Guidance_Classifier
import matplotlib.pyplot as plt
import numpy as np
import random
from ..base.base_utils import CPU_Unpickler


from .utils import N_log_prob

# parent class
class BetaVAEgp(VAE):
    r"""
    :math:`\beta`-VAE model.

    Args:
        model_config (BetaVAEgpConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        notation:
        input contains data_x, data_y, data_t, ..., splits
        L: dimension of input dimension x
        D: dimension of latent space z
        N_patients: number of patients (per batch)
        T_i: number of samples of patient i (in batch)
    """

    def __init__(
        self,
        model_config: BetaVAEgpConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        classifiers=None,
        mu_predictor=None,
        var_predictor=None,
    ):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "BetaVAEgp"
        self.predict = model_config.predict
        self.progression = model_config.progression

        self.latent_dim = model_config.latent_dim
        self.missing_loss = model_config.missing_loss
        self.model_config = model_config
        self.beta = model_config.beta
        self.w_recon = model_config.w_recon
        self.w_class = model_config.w_class

        self.w_recon_pred = model_config.w_recon_pred
        self.w_class_pred = model_config.w_class_pred

        self.splits_x0 = model_config.splits_x0
        self.kinds_x0 = model_config.kinds_x0

        self.splits_y0 = model_config.splits_y0
        self.kinds_y0 = model_config.kinds_y0

        self.weights_x0 = model_config.weights_x0
        self.weights_y0 = model_config.weights_y0

        self.to_reconstruct_x = model_config.to_reconstruct_x
        self.to_reconstruct_y = model_config.to_reconstruct_y

        self.classifiers = classifiers
        self.mu_predictor = mu_predictor
        self.var_predictor = var_predictor
        # if cuda is available, transfer classifier to GPU
        if (not classifiers is None) and torch.cuda.is_available():
            self.classifiers = [classifier.cuda() for classifier in self.classifiers]

    def forward(self, inputs: MissingDataset, **kwargs):
        """
        input, encoding, sampling, decoding, guidance, loss
        """

        ################################
        ### 1) INPUT
        ################################

        # input data (N_patients x T_i) x L
        data_x = inputs["data_x"]

        # non-missing entries in the input/output data
        if self.missing_loss:
            non_missing_x = 1 - inputs["missing_x"] * 1.0
        else:
            non_missing_x = 1

        splits = inputs["splits"]  # N_patients
        # times = inputs["data_t"][:, 0].reshape(-1, 1)  # N_patients x 1
        times = inputs["data_x"][:, 0].reshape(-1, 1)  # N_patients x 1

        if not self.classifiers is None:
            data_y = inputs["data_y"]  # N_patients x n_class
            non_missing_y = 1 - inputs["missing_y"] * 1.0  # N_patients x n_class
        else:
            data_y = 0
            non_missing_y = 0

        ################################
        ### 2) ENCODING
        ################################

        # encode the input into dimension D
        # mu: (N_patients x T_i) x D
        # log_var: (N_patients x T_i) x D
        if self.predict:
            # prepare input for prediction (i.e. to predict ...-th sample of each patient)

            (
                data_x_recon,
                data_y_recon,
                non_missing_x_recon,
                non_missing_y_recon,
                data_x_pred,
                data_y_pred,
                non_missing_x_pred,
                non_missing_y_pred,
                times_recon,
                times_pred,
                times_pred_grouped,
                splits_flattened,
                num_for_rec,
                num_for_rec_flat,
                num_to_pred,
                num_to_pred_flat,
                num_to_rec_num_to_pred,
            ) = self.get_splits_for_prediction(
                data_x, data_y, non_missing_x, non_missing_y, splits, times
            )
        else:
            data_x_recon = data_x
            data_y_recon = data_y
            non_missing_x_recon = non_missing_x
            non_missing_y_recon = non_missing_y
            splits_flattened = splits
            times_recon = times
            times_pred = None
            num_for_rec = None
            data_x_pred = None
            data_y_pred = None
            non_missing_x_pred = None
            non_missing_y_pred = None
            num_for_rec = None
            num_for_rec_flat = splits
            num_to_pred = None
            num_to_pred_flat = [0 for i in range(len(splits))]
            num_to_rec_num_to_pred = splits

        encoder_output = self.encoder(data_x_recon)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        if self.predict:
            # predicted mu, log_var. mu_reassembled is the concatenation of mu and mu_pred for each patient
            (
                mu_pred,
                log_var_pred,
                mu_reassembled,
                log_var_reassembled,
            ) = self.predict_latent(
                mu,
                log_var,
                splits,
                num_for_rec_flat,
                num_to_pred_flat,
                times_pred_grouped,
            )

        else:
            mu_reassembled = mu
            log_var_reassembled = log_var
            mu_pred = None
            log_var_pred = None

        ################################
        ### 3) SAMPLING
        ################################

        # sample from the latent distribution
        # z: (N_patients x T_i) x D

        z, kwargs = self.sample(mu_reassembled, log_var_reassembled)
        # split per patient
        # splits_flattened = num_to_rec_num_to_pred

        z_splitted = torch.split(z, num_to_rec_num_to_pred, dim=0)

        ################################
        ### 4a) DECODING
        ################################
        if self.predict:
            # only for loss tracking purposes, I guess we could keep them as one
            z_recon = torch.cat(
                [
                    z_splitted[index][: -num_to_pred_flat[index], :]
                    for index, num_v in enumerate(num_to_rec_num_to_pred)
                    if num_v > 0
                ]
            )
            z_pred = torch.cat(
                [
                    z_splitted[index][-num_to_pred_flat[index] :, :]
                    for index, num_v in enumerate(num_to_rec_num_to_pred)
                    if num_v > 0
                ]
            )
            # concatenate with time
            z_recon = torch.cat([z_recon, times_recon], axis=1)
            z_pred = torch.cat([z_pred, times_pred], axis=1)
            mu = torch.cat([mu, times_recon], axis=1)
            mu_pred = torch.cat([mu_pred, times_pred], axis=1)
            recon_x_pred = self.decoder(z_pred)["reconstruction"]
            recon_m_pred = self.decoder(mu_pred)["reconstruction"]

        else:
            z_recon = torch.cat([z, times_recon], axis=1)
            mu = torch.cat([mu, times_recon], axis=1)
            z_pred = None
            recon_x_pred = None
            recon_m_pred = None

        # reconstruction of the sample z
        # recon: (N_patients x T_i) x L
        recon_x = self.decoder(z_recon)["reconstruction"]  # sample
        recon_m = self.decoder(mu)["reconstruction"]  # NN output

        recon_x_log_var = self.decoder(z_recon)["reconstruction_log_var"]  # sample
        recon_m_log_var = self.decoder(mu)["reconstruction_log_var"]  # NN output

        # print()

        ## x) samlping from the output?

        ################################
        ### 4b) GUIDANCE
        ################################
        # reconstruction labels
        y_recon_out, y_recon_out_m = torch.zeros_like(
            data_y_recon, device=self.device
        ), torch.zeros_like(data_y_recon, device=self.device)
        if self.predict:
            # prediction labels
            y_pred_out, y_m_pred_out = torch.zeros_like(
                data_y_pred, device=self.device
            ), torch.zeros_like(data_y_pred, device=self.device)
        else:
            y_m_pred_out, y_pred_out = None, None

        for classifier in self.classifiers:
            if classifier.type == "static":
                y_recon_out[:, classifier.y_dims] = classifier.forward(
                    z_recon[:, classifier.z_dims]
                )
                y_recon_out_m[:, classifier.y_dims] = classifier.forward(
                    mu[:, classifier.z_dims]
                )
                if self.predict:
                    y_pred_out[:, classifier.y_dims] = classifier.forward(
                        z_pred[:, classifier.z_dims]
                    )
                    y_m_pred_out[:, classifier.y_dims] = classifier.forward(
                        mu_pred[:, classifier.z_dims]
                    )

        if self.progression:
            if self.predict:
                raise NotImplementedError(
                    "Progression classifier not implemented for prediction"
                )
            # for the progression classifiers, reshape z to have the accurate input shape/content
            z_splitted_stacked = torch.cat(
                [
                    torch.stack(
                        [torch.zeros((2, z.shape[1]), device=self.device)]
                        + [z_splitted[index][i : i + 2] for i in range(0, num_v - 1)],
                        dim=0,
                    )
                    for index, num_v in enumerate(num_to_rec_num_to_pred)
                    if num_v > 0
                ]
            )

            mu_splitted = torch.split(mu_reassembled, num_to_rec_num_to_pred, dim=0)
            mu_splitted_stacked = torch.cat(
                [
                    torch.stack(
                        [torch.zeros((2, mu_reassembled.shape[1]), device=self.device)]
                        + [mu_splitted[index][i : i + 2] for i in range(0, num_v - 1)],
                        dim=0,
                    )
                    for index, num_v in enumerate(num_to_rec_num_to_pred)
                    if num_v > 0
                ]
            )
            for classifier in self.classifiers:
                if classifier.type == "progression":
                    # progression classifiers
                    y_recon_out[:, classifier.y_dims] = classifier.forward(
                        z_splitted_stacked[:, :, classifier.z_dims].reshape(
                            z_splitted_stacked[:, :, classifier.z_dims].shape[0],
                            classifier.input_dim,
                        )
                    )
                    y_recon_out_m[:, classifier.y_dims] = classifier.forward(
                        mu_splitted_stacked[:, :, classifier.z_dims].reshape(
                            mu_splitted_stacked[:, :, classifier.z_dims].shape[0],
                            classifier.input_dim,
                        )
                    )

        ################################
        ### 5) LOSS
        ################################

        loss_recon, loss_recon_stacked = self.loss_reconstruct(
            recon_x, data_x_recon, non_missing_x_recon, output_log_var=recon_x_log_var
        )
        if self.predict:
            loss_recon_pred, loss_recon_stacked_pred = self.loss_reconstruct(
                recon_x_pred, data_x_pred, non_missing_x_pred
            )
        else:
            loss_recon_pred, loss_recon_stacked_pred = torch.zeros(
                1, device=self.device
            ), torch.zeros(1, device=self.device)

        loss_kld, kwargs_output = self.loss_kld(
            mu_reassembled, log_var_reassembled, **kwargs
        )
        # loss_kld, kwargs_output = self.loss_kld(mu, log_var, times, splits, **kwargs)
        # for pred also ?
        # different weights per classifier

        loss_class_weighted = torch.zeros(len(self.classifiers), device=self.device)
        loss_class = torch.zeros(len(self.classifiers), device=self.device)
        loss_class_weighted_pred = torch.zeros(
            len(self.classifiers), device=self.device
        )
        loss_class_pred = torch.zeros(len(self.classifiers), device=self.device)
        for i, classifier in enumerate(self.classifiers):
            loss_class[i] = self.loss_classifier(
                y_recon_out[:, classifier.y_dims],
                data_y_recon[:, classifier.y_dims],
                non_missing_y_recon[:, classifier.y_dims],
                [self.splits_y0[elem] for elem in classifier.y_indices],
                [self.kinds_y0[elem] for elem in classifier.y_indices],
            )[0]
            loss_class_weighted[i] = self.w_class[classifier.name] * loss_class[i]
            if self.predict:
                loss_class_pred[i] = self.loss_classifier(
                    y_pred_out[:, classifier.y_dims],
                    data_y_pred[:, classifier.y_dims],
                    non_missing_y_pred[:, classifier.y_dims],
                    [self.splits_y0[elem] for elem in classifier.y_indices],
                    [self.kinds_y0[elem] for elem in classifier.y_indices],
                )[0]
                loss_class_weighted_pred[i] = (
                    self.w_class_pred[classifier.name] * loss_class_pred[i]
                )
        if self.predict:
            loss_recon_stacked_mean = torch.cat(
                [loss_recon_stacked, loss_recon_stacked_pred]
            ).mean()
        else:
            loss_recon_stacked_mean = loss_recon_stacked.mean()
        loss = (
            self.w_recon * loss_recon_stacked_mean
            + self.beta * loss_kld
            + torch.sum(loss_class_weighted)
        )

        loss_pred_unw = loss_recon_pred + torch.sum(loss_class_pred)
        loss_pred = self.w_recon_pred * loss_recon_pred + torch.sum(
            loss_class_weighted_pred
        )
        loss += torch.sum(loss_class_weighted_pred)

        losses_unweighted = torch.cat(
            [
                loss_recon.reshape(-1, 1),
                loss_kld.reshape(-1, 1),
                loss_class.reshape(-1, 1),
            ]
        )
        losses_weighted = torch.cat(
            [
                self.w_recon * loss_recon.reshape(-1, 1),
                self.beta * loss_kld.reshape(-1, 1),
                loss_class_weighted.reshape(-1, 1),
            ],
        )
        losses_unweighted_pred = torch.cat(
            [
                loss_recon_pred.reshape(-1, 1),
                loss_class_pred.reshape(-1, 1),
            ]
        )
        losses_weighted_pred = torch.cat(
            [
                self.w_recon_pred * loss_recon_pred.reshape(-1, 1),
                loss_class_weighted_pred.reshape(-1, 1),
            ],
        )

        ################################
        ### 6) OUTPUT
        ################################
        if self.predict:
            y_recon_out_splitted = torch.split(y_recon_out, num_for_rec_flat, dim=0)
            y_pred_out_splitted = torch.split(y_pred_out, num_to_pred_flat, dim=0)
            y_all = torch.cat(
                [
                    torch.cat([y_recon_out_splitted[index], y_pred_out_splitted[index]])
                    for index in range(len(y_pred_out_splitted))
                ]
            )
        else:
            y_all = y_recon_out
        output = ModelOutput(
            loss_recon=self.w_recon * loss_recon,
            loss_kld=self.beta * loss_kld,
            loss_class=torch.sum(loss_class_weighted),
            losses_unweighted=losses_unweighted,
            losses_weighted=losses_weighted,
            losses_weighted_pred=losses_weighted_pred,
            losses_unweighted_pred=losses_unweighted_pred,
            loss_recon_stacked=loss_recon_stacked,
            loss=loss,
            loss_pred=loss_pred,
            loss_pred_unw=loss_pred_unw,
            loss_recon_stacked_pred=loss_recon_stacked_pred,
            recon_x=recon_x,
            recon_m=recon_m,
            recon_x_log_var=recon_x_log_var,
            recon_m_log_var=recon_m_log_var,
            z=z,
            z_recon=z_recon,
            z_pred=z_pred,
            mu=mu,
            mu_pred=mu_pred,
            log_var_pred=log_var_pred,
            y_out_pred=y_pred_out,
            y_out_m_pred=y_m_pred_out,
            data_x_pred=data_x_pred,
            data_y_pred=data_y_pred,
            non_missing_x_pred=non_missing_x_pred,
            non_missing_y_pred=non_missing_y_pred,
            recon_x_pred=recon_x_pred,
            recon_m_pred=recon_m_pred,
            log_var=log_var,
            y_out_rec=y_recon_out,
            y_out_m_rec=y_recon_out_m,
            y_all=y_all,
        )

        # update output with subclass specific values
        for key, values in kwargs_output.items():
            output[key] = values

        return output

    """
    # old version with only mse
    def loss_recon(self, recon_x, x, non_missing_x):


        #if self.model_config.reconstruction_loss == "mse":

        # set loss for missing dimensions to 0!
        recon_loss = ( non_missing_x *
            F.mse_loss(
            recon_x,
            x,
            reduction="none",
        )).sum(dim=-1)

        return recon_loss.mean(dim=0)
    """

    def predict_latent(
        self, mu, log_var, splits, num_for_rec, num_to_pred, times_pred_grouped
    ):
        (
            mu_for_pred,
            log_var_for_pred,
            mu_for_recon_splitted,
            log_var_for_recon_splitted,
            times_extended,
        ) = self.reshape_input_for_pred(
            mu,
            log_var,
            splits,
            num_for_rec,
            num_to_pred,
            times_pred_grouped,
            lstm_predictor=self.model_config.predictor_config.lstm_predictor,
        )
        # predict mu_T and log_var_T
        mu_pred = self.mu_predictor(mu_for_pred, times_extended).out
        log_var_pred = self.var_predictor(log_var_for_pred, times_extended).out
        #  assemble mu_1,...,T-1 with mu_T,..mu_T + max_num_pred for each patient
        mu_pred_splitted = torch.split(mu_pred, num_to_pred, dim=0)
        log_var_pred_splitted = torch.split(log_var_pred, num_to_pred, dim=0)
        mu_reassembled = torch.cat(
            [
                torch.cat([mu_for_recon_splitted[index], mu_pred_splitted[index]])
                for index in range(len(mu_pred_splitted))
            ]
        )
        log_var_reassembled = torch.cat(
            [
                torch.cat(
                    [
                        log_var_for_recon_splitted[index],
                        log_var_pred_splitted[index],
                    ]
                )
                for index in range(len(log_var_pred_splitted))
            ]
        )
        return mu_pred, log_var_pred, mu_reassembled, log_var_reassembled

    def get_splits_for_prediction(
        self, data_x, data_y, non_missing_x, non_missing_y, splits, times
    ):
        max_num_to_pred = self.model_config.max_num_to_pred
        # num_for_rec: nested list of length N_patients, each sublist contains the number of samples used for reconstruction
        if self.model_config.num_for_rec == "random":
            raise NotImplementedError(
                "Random reconstruction not implemented yet, please use 'all'"
            )
            # for each patient, predict a random sample, and use the previous samples for reconstruction
            num_for_rec = [
                [random.randint(1, elem - 1)] if elem > 1 else [0] for elem in splits
            ]
        elif self.model_config.num_for_rec == "last":
            # raise NotImplementedError("Last reconstruction not implemented yet, please use 'all'")
            # predict last min(num_samples -1, max_num_to_pred) samples

            num_for_rec = [
                [i for i in range(max(1, elem - max_num_to_pred), elem)]
                for elem in splits
            ]
            num_to_pred = [
                [
                    min(max_num_to_pred, elem - i)
                    for i in range(max(1, elem - max_num_to_pred), elem)
                ]
                for elem in splits
            ]

        elif self.model_config.num_for_rec == "all":
            # for each patient, for each sample from 1 to T_i, use all samples up to that sample for reconstruction
            # and try to predict up to max_num_to_pred samples
            num_for_rec = [list(range(1, elem)) if elem > 1 else [0] for elem in splits]
            # number of reconstructed samples
            # num_for_rec_flat = [elem for sublist in num_for_rec for elem in sublist]
            num_to_pred = [
                [min(max_num_to_pred, num_v - i) for i in range(1, num_v)]
                for pat, num_v in enumerate(splits)
                if num_v > 0
            ]
            # number of predicted samples
            # num_to_pred_flat = [elem for sublist in num_to_pred for elem in sublist]
            # # sum of reconstructed and predicted samples
            # num_to_rec_num_to_pred = list(np.array(num_for_rec_flat) + np.array(num_to_pred_flat))
        # number of reconstructed samples
        num_for_rec_flat = [elem for sublist in num_for_rec for elem in sublist]
        # number of predicted samples
        num_to_pred_flat = [elem for sublist in num_to_pred for elem in sublist]
        # sum of reconstructed and predicted samples
        num_to_rec_num_to_pred = list(
            np.array(num_for_rec_flat) + np.array(num_to_pred_flat)
        )
        # contains the new data splits
        splits_flattened = [item + 1 for sublist in num_for_rec for item in sublist]
        (
            data_x_recon,
            data_y_recon,
            non_missing_x_recon,
            non_missing_y_recon,
            data_x_pred,
            data_y_pred,
            non_missing_x_pred,
            non_missing_y_pred,
            times_recon,
            times_pred,
            times_pred_grouped,
        ) = self.reshape_x_y_input(
            data_x,
            data_y,
            non_missing_x,
            non_missing_y,
            splits,
            num_for_rec,
            times,
            max_num_to_pred,
        )
        return (
            data_x_recon,
            data_y_recon,
            non_missing_x_recon,
            non_missing_y_recon,
            data_x_pred,
            data_y_pred,
            non_missing_x_pred,
            non_missing_y_pred,
            times_recon,
            times_pred,
            times_pred_grouped,
            splits_flattened,
            num_for_rec,
            num_for_rec_flat,
            num_to_pred,
            num_to_pred_flat,
            num_to_rec_num_to_pred,
        )

    def reshape_x_y_input(
        self,
        data_x,
        data_y,
        non_missing_x,
        non_missing_y,
        splits,
        num_for_rec,
        times,
        max_num_to_pred,
    ):
        data_x_splitted = torch.split(data_x, splits, dim=0)
        times_splitted = torch.split(times, splits, dim=0)
        # reconstruction ground truth data
        # iterate over patients and list of number of reconstruction inputs per patient
        data_x_recon = torch.cat(
            [
                data_x_splitted[pat][: num_for_rec[pat][vis], :]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        # prediction ground truth data
        data_x_pred = torch.cat(
            [
                data_x_splitted[pat][
                    num_for_rec[pat][vis] : num_for_rec[pat][vis] + max_num_to_pred, :
                ]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        data_y_splitted = torch.split(data_y, splits, dim=0)
        # labels reconstruction ground truth
        data_y_recon = torch.cat(
            [
                data_y_splitted[pat][: num_for_rec[pat][vis], :]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        # labels prediction ground truth
        data_y_pred = torch.cat(
            [
                data_y_splitted[pat][
                    num_for_rec[pat][vis] : num_for_rec[pat][vis] + max_num_to_pred, :
                ]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        non_missing_x_splitted = torch.split(non_missing_x, splits, dim=0)
        # non missing data x reconstruction ground truth
        non_missing_x_recon = torch.cat(
            [
                non_missing_x_splitted[pat][: num_for_rec[pat][vis], :]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        non_missing_x_pred = torch.cat(
            [
                non_missing_x_splitted[pat][
                    num_for_rec[pat][vis] : num_for_rec[pat][vis] + max_num_to_pred, :
                ]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        non_missing_y_splitted = torch.split(non_missing_y, splits, dim=0)
        non_missing_y_recon = torch.cat(
            [
                non_missing_y_splitted[pat][: num_for_rec[pat][vis], :]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        non_missing_y_pred = torch.cat(
            [
                non_missing_y_splitted[pat][
                    num_for_rec[pat][vis] : num_for_rec[pat][vis] + max_num_to_pred, :
                ]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        # times for later concatenation
        times_recon = torch.cat(
            [
                times_splitted[pat][: num_for_rec[pat][vis], :]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        times_pred = torch.cat(
            [
                times_splitted[pat][
                    num_for_rec[pat][vis] : num_for_rec[pat][vis] + max_num_to_pred, :
                ]
                for pat, num_v in enumerate(splits)
                for vis in range(len(num_for_rec[pat]))
                if num_v > 0
            ]
        )
        # times_pred_grouped = [[times_splitted[pat][i:i+max_num_to_pred] for i in range(1, num_v)] for pat, num_v in enumerate(splits) if num_v > 0]
        times_pred_grouped = [
            [
                times_splitted[pat][
                    num_for_rec[pat][vis] : num_for_rec[pat][vis] + max_num_to_pred, :
                ]
                for vis in range(len(num_for_rec[pat]))
            ]
            for pat, num_v in enumerate(splits)
            if num_v > 0
        ]

        return (
            data_x_recon,
            data_y_recon,
            non_missing_x_recon,
            non_missing_y_recon,
            data_x_pred,
            data_y_pred,
            non_missing_x_pred,
            non_missing_y_pred,
            times_recon,
            times_pred,
            times_pred_grouped,
        )

    def reshape_input_for_pred(
        self,
        mu,
        log_var,
        splits,
        num_for_rec,
        num_to_pred,
        times_pred_grouped,
        lstm_predictor=False,
    ):
        # splits_for_pred = [
        #     elem[vis]
        #     for index, elem in enumerate(num_for_rec)
        #     for vis in range(len(elem))
        #     if splits[index] > 0
        # ]
        mu_splitted = torch.split(mu, num_for_rec, dim=0)
        log_var_splitted = torch.split(log_var, num_for_rec, dim=0)
        mu_splitted_extended = [
            [mu_splitted[i]] * elem for i, elem in enumerate(num_to_pred)
        ]
        mu_splitted_extended = [
            item for sublist in mu_splitted_extended for item in sublist
        ]
        log_var_splitted_extended = [
            [log_var_splitted[i]] * elem for i, elem in enumerate(num_to_pred)
        ]
        log_var_splitted_extended = [
            item for sublist in log_var_splitted_extended for item in sublist
        ]
        lengths = [len(elem) for elem in mu_splitted_extended]
        times_extended = torch.cat([torch.cat(elem) for elem in times_pred_grouped])
        if lstm_predictor:
            mu_splitted_stacked = torch.nn.utils.rnn.pad_sequence(
                mu_splitted_extended, batch_first=True
            )
            # lengths = [elem for elem in splits_for_pred if elem > 0]
            mu_splitted_stacked = torch.nn.utils.rnn.pack_padded_sequence(
                mu_splitted_stacked,
                batch_first=True,
                lengths=lengths,
                enforce_sorted=False,
            )
            log_var_splitted_stacked = torch.nn.utils.rnn.pad_sequence(
                log_var_splitted_extended, batch_first=True
            )
            log_var_splitted_stacked = torch.nn.utils.rnn.pack_padded_sequence(
                log_var_splitted_stacked,
                batch_first=True,
                lengths=lengths,
                enforce_sorted=False,
            )

        else:
            n_previous_for_pred = self.model_config.predictor_config.n_previous_for_pred
            # discard first label since its not predicted

            mu_splitted_stacked = [
                F.pad(mu_splitted[pat], (0, 0, n_previous_for_pred, 0))
                for pat, num_v in enumerate(splits)
                if num_v > 0
            ]

            mu_splitted_stacked = torch.cat(
                [
                    torch.stack(
                        [
                            elem[i : i + n_previous_for_pred].flatten()
                            for i in range(len(elem) - n_previous_for_pred)
                        ],
                        dim=0,
                    )
                    for elem in mu_splitted_stacked
                ]
            )
            log_var_splitted_stacked = [
                F.pad(log_var_splitted[pat], (0, 0, n_previous_for_pred, 0))
                for pat, num_v in enumerate(splits)
                if num_v > 0
            ]
            log_var_splitted_stacked = torch.cat(
                [
                    torch.stack(
                        [
                            elem[i : i + n_previous_for_pred].flatten()
                            for i in range(len(elem) - n_previous_for_pred)
                        ],
                        dim=0,
                    )
                    for elem in log_var_splitted_stacked
                ]
            )

        return (
            mu_splitted_stacked,
            log_var_splitted_stacked,
            mu_splitted,
            log_var_splitted,
            times_extended,
        )

    def loss_multioutput(
        self,
        output,
        target,
        non_missing,
        splits0,
        kinds0,
        weights0,
        output_log_var=None,
        to_reconstruct=[],
    ):
        """
        general multi-output loss with missing data
        """
        # if to_reconstruct is empty list, populate with true for all
        if len(to_reconstruct) == 0:
            raise ValueError(
                "to_reconstruct must be a list of tuples (var_name, var_index, bool)"
            )

        # split the inputs into list of each dimension/variables
        # (i.e. for categorical variables we get a tensor N x n_categories
        # and for binary and conitionsou N x 1)
        output_splitted = torch.split(output, splits0, dim=1)
        target_splitted = torch.split(target, splits0, dim=1)
        non_missing_splitted = torch.split(non_missing, splits0, dim=1)

        if not output_log_var is None:
            output_log_var_splitted = torch.split(output_log_var, splits0, dim=1)
        else:
            output_log_var_splitted = None

        # loop over the output dimensions
        loss_state = []
        # save the NLL and CE losses separately
        nll_loss_state = []
        ce_loss_state = []
        for i, out in enumerate(output_splitted):
            nll_loss = None
            ce_loss = None

            if (kinds0[i] == "continuous") or (kinds0[i] == "ordinal"):

                # dimension_loss = F.mse_loss(out, target_splitted[i], reduction="none")

                if output_log_var is None:

                    dimension_loss = F.mse_loss(
                        out, target_splitted[i], reduction="none"
                    )
                    nll_loss = dimension_loss

                else:
                    if to_reconstruct[i][2]:
                        dimension_loss = N_log_prob(
                            out, output_log_var_splitted[i], target_splitted[i]
                        )
                        nll_loss = dimension_loss
                    else:
                        dimension_loss = torch.zeros_like(out, device=self.device)

                    # dimension_loss = N_log_prob(out, torch.zeros_like(out), target_splitted[i]) # same as MSE

            elif kinds0[i] == "categorical":
                if to_reconstruct[i][2]:
                    # dimension_loss = F.cross_entropy(
                    #     out,
                    #     target_splitted[i],
                    #     weight=torch.tensor(weights0[i], device=self.device),
                    #     reduction="none",
                    # ).unsqueeze(1)
                    dimension_loss = F.cross_entropy(
                        out,
                        target_splitted[i],
                        reduction="none",
                    ).unsqueeze(1)
                    ce_loss = dimension_loss
                else:
                    dimension_loss = torch.zeros_like(out, device=self.device)
                # dimension_loss = F.cross_entropy(
                #     out, target_splitted[i], reduction="none"
                # ).unsqueeze(1)

            elif kinds0[i] == "binary":
                if to_reconstruct[i][2]:
                    # dimension_loss = F.binary_cross_entropy(
                    #     torch.sigmoid(out), target_splitted[i], reduction="none"
                    # )
                    # dimension_loss = F.binary_cross_entropy_with_logits(
                    #     out,
                    #     target_splitted[i],
                    #     reduction="none",
                    #     pos_weight=torch.tensor(weights0[i], device=self.device),
                    # )
                    dimension_loss = F.binary_cross_entropy_with_logits(
                        out,
                        target_splitted[i],
                        reduction="none",
                    )
                    ce_loss = dimension_loss
                else:
                    dimension_loss = torch.zeros_like(out, device=self.device)
                # dimension_loss = F.binary_cross_entropy(
                #     torch.sigmoid(out), target_splitted[i], reduction="none"
                # )

            else:
                print("loss not implemented")

            # take only the contribution where we have observations
            dimension_loss = dimension_loss[non_missing_splitted[i][:, 0] == 1.0, :]

            # average over all observed samples
            dimension_loss = dimension_loss.mean()  # dim=0)
            if ce_loss is not None:
                ce_loss = ce_loss[non_missing_splitted[i][:, 0] == 1.0, :]
                ce_loss = ce_loss.mean()
                if ce_loss.isnan():
                    ce_loss = torch.zeros_like(ce_loss)
                ce_loss_state.append(ce_loss)
            if nll_loss is not None:
                nll_loss = nll_loss[non_missing_splitted[i][:, 0] == 1.0, :]
                nll_loss = nll_loss.mean()
                if nll_loss.isnan():
                    nll_loss = torch.zeros_like(nll_loss)
                nll_loss_state.append(nll_loss)

            if (
                dimension_loss.isnan()
            ):  # if we have no observation, then the mean produces nan!
                dimension_loss = torch.zeros_like(dimension_loss)

            loss_state.append(dimension_loss)

        # stack it together and sum all dimensions up
        loss_state_stacked = torch.stack(loss_state, dim=0)
        nll_loss = torch.zeros(1).to(self.device)
        if len(nll_loss_state) > 0:
            nll_loss_state_stacked = torch.stack(nll_loss_state, dim=0)
            nll_loss = nll_loss_state_stacked.mean(dim=-1)
        ce_loss = torch.zeros(1).to(self.device)
        if len(ce_loss_state) > 0:
            ce_loss_state_stacked = torch.stack(ce_loss_state, dim=0)
            ce_loss = ce_loss_state_stacked.mean(dim=-1)
        # print(loss_state.sum(dim=-1), loss_state)

        # print(loss_state.sum(dim=-1), loss_state)
        # changed to mean
        loss_state = loss_state_stacked.mean(dim=-1)

        return loss_state, loss_state_stacked, nll_loss, ce_loss

    def loss_reconstruct(self, recon_x, x, non_missing_x, output_log_var=None):
        return self.loss_multioutput(
            recon_x,
            x,
            non_missing_x,
            self.splits_x0,
            self.kinds_x0,
            self.weights_x0,
            output_log_var=output_log_var,
            to_reconstruct=self.to_reconstruct_x,
        )

    def loss_classifier(self, y_out, y_true, non_missing_y, splits, kinds, weights):
        """
        compute the classification loss
        """

        if self.classifiers is None:
            ll = torch.zeros(1).to(self.device)

        else:
            # replace the missing values with the true values
            # miss = missing_y == 1.
            # y_out_miss = y_out.clone()
            # y_out_miss[miss] = y_true[miss].type(torch.float)

            # cross-entropy loss -> put it in init or input
            # loss = torch.nn.CrossEntropyLoss(reduction='mean')

            # compute the loss only if we have a lable
            # ll = loss( y_out_miss, y_true.type(torch.float) )

            ll = self.loss_multioutput(
                y_out,
                y_true.type(torch.float),
                non_missing_y,
                splits,
                kinds,
                weights,
                output_log_var=None,
                to_reconstruct=self.to_reconstruct_y,
            )

        return ll

    def classifier_forward(self, mu):
        """
        guidance classifier forward
        """

        if self.classifier is None:
            y_out = torch.zeros(1).to(self.device)

        else:
            y_out = self.classifier.forward(mu)

        return y_out

    def sample(self, mu, *args):
        return mu, {}

    def loss_kld(self, *args, **kwargs):
        return torch.zeros(1).to(self.device), {}

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        # Sample N(mu, var)
        return mu + eps * std, eps

    # TODO move plots somewhere else
    def plot_z(self, z, mu, log_var, time, figsize=(6, 3), CI=True, MUplot=True):
        z = z.detach().numpy()
        mu = mu.detach().numpy()
        log_var = log_var.detach().numpy()

        nP = z.shape[1]

        # create figure with subplots
        f, axs = plt.subplots(
            nP, 1, sharex=True, sharey=False, figsize=(1 * figsize[0], nP * figsize[1])
        )
        f.subplots_adjust(hspace=0.2)  # , wspace=0.2)

        for i in range(nP):
            if nP > 1:
                ax = axs[i]
            else:
                ax = axs
            ax.plot(time, z[:, i], ".:", color="C2", label="z")
            if MUplot:
                ax.plot(time, mu[:, i], ".-", color="C0", label="mu")

            if CI and MUplot:
                CIadd = 1.96 * np.sqrt(np.exp(log_var))
                ax.fill_between(
                    time,
                    mu[:, i] - CIadd[:, i],
                    mu[:, i] + CIadd[:, i],
                    color="C" + str(i),
                    alpha=0.1,
                )

            # ax.set_ylim(-2,2)
            ax.legend()

        return axs

    def plot_x(
        self,
        data_x,
        recon_x,
        recon_m,
        time,
        missing_x,
        names_x=None,
        probs_matrix=None,
        probs_matrix_m=None,
        kinds_x1=None,
        recon_m_log_var=None,
        figsize=(6, 3),
    ):
        recon_x = recon_x.detach().numpy()
        recon_m = recon_m.detach().numpy()
        if not probs_matrix is None:
            probs_matrix = probs_matrix.detach().numpy()
        if not probs_matrix_m is None:
            probs_matrix_m = probs_matrix_m.detach().numpy()
        if not recon_m_log_var is None:
            recon_m_log_var = recon_m_log_var.detach().numpy()

        nP = data_x.shape[1]

        # create figure with subplots
        f, axs = plt.subplots(
            nP, 1, sharex=True, sharey=False, figsize=(1 * figsize[0], nP * figsize[1])
        )
        f.subplots_adjust(hspace=0.2)  # , wspace=0.2)

        for i in range(nP):
            if nP > 1:
                ax = axs[i]
            else:
                ax = axs
            ax.plot(time, data_x[:, i], ".-", color="C2", label="data_x")

            if kinds_x1 is None:
                ax.plot(time, recon_x[:, i], ".:", color="C0", label="recon_x")
                ax.plot(time, recon_m[:, i], ".-", color="C0", label="recon_m")

                if not probs_matrix is None:
                    ax.plot(time, probs_matrix[:, i], ".:", color="C1", label="probs")
                if not probs_matrix_m is None:
                    ax.plot(
                        time, probs_matrix_m[:, i], ".-", color="C1", label="probs_m"
                    )
            else:
                if kinds_x1[i] in ["continuous", "ordinal"]:
                    ax.plot(time, recon_x[:, i], ".:", color="C0", label="recon_x")
                    ax.plot(time, recon_m[:, i], ".-", color="C0", label="recon_m")

                    if not recon_m_log_var is None:
                        CIa = 1.96 * np.exp(0.5 * recon_m_log_var[:, i])
                        ax.fill_between(
                            time,
                            recon_m[:, i] - CIa,
                            recon_m[:, i] + CIa,
                            alpha=0.1,
                            color="C0",
                            label="CI_recon_m",
                        )

                        ax.set_ylim(-4, 4)
                else:
                    if not probs_matrix is None:
                        ax.plot(
                            time, probs_matrix[:, i], ".:", color="C1", label="probs"
                        )
                    if not probs_matrix_m is None:
                        ax.plot(
                            time,
                            probs_matrix_m[:, i],
                            ".-",
                            color="C1",
                            label="probs_m",
                        )
                    ax.set_ylim(-1.5, 1.5)

            non_miss = ~missing_x[:, i]
            ax.plot(
                time[non_miss],
                data_x[non_miss, i],
                "o",
                color="C3",
                label="non-missing_x",
            )

            if not names_x is None:
                ax.set_title(names_x[i])

            # ax.set_ylim(-3, 3)
            ax.legend()

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None
        my_classifiers = cls._load_custom_modules_from_folder(
            dir_path, "classifiers.pkl"
        )
        my_mu_predictor = cls._load_custom_modules_from_folder(
            dir_path, "mu_predictor.pkl"
        )
        my_var_predictor = cls._load_custom_modules_from_folder(
            dir_path, "var_predictor.pkl"
        )

        model = cls(
            model_config,
            encoder=encoder,
            decoder=decoder,
            classifiers=my_classifiers,
            mu_predictor=my_mu_predictor,
            var_predictor=my_var_predictor,
        )
        model.load_state_dict(model_weights)

        return model

    @classmethod
    def _load_custom_modules_from_folder(cls, dir_path, file_name):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if file_name not in file_list:
            raise FileNotFoundError(
                f"Missing file {file_name} in"
                f"{dir_path}..."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, file_name), "rb") as fp:
                modules = CPU_Unpickler(fp).load()

        return modules

    def save(self, dir_path: str):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        env_spec = EnvironmentConfig(
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
        )
        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except FileNotFoundError as e:
                raise e

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoder))
                cloudpickle.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoder))
                cloudpickle.dump(self.decoder, fp)

        for classif in self.classifiers:
            cloudpickle.register_pickle_by_value(inspect.getmodule(classif))
        with open(os.path.join(dir_path, "mu_predictor.pkl"), "wb") as fp:
            cloudpickle.dump(self.mu_predictor, fp)
        with open(os.path.join(dir_path, "var_predictor.pkl"), "wb") as fp:
            cloudpickle.dump(self.var_predictor, fp)

        with open(os.path.join(dir_path, "classifiers.pkl"), "wb") as fp:
            cloudpickle.dump(self.classifiers, fp)

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))


class BetaVAEgpInd(BetaVAEgp):
    """
    This version correspond to the usual beta VAE with independent states and a guidance classifier.
    """

    def __init__(
        self,
        model_config: BetaVAEgpConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        classifiers=None,
        mu_predictor=None,
        var_predictor=None,
    ):
        super().__init__(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
            classifiers=classifiers,
            mu_predictor=mu_predictor,
            var_predictor=var_predictor,
        )

        self.model_name = "BetaVAEgpInd"

        self.latent_prior_noise_var = model_config.latent_prior_noise_var

    def sample(self, mu, log_var, *args):
        # sample once from N(mu, var)
        # z: (N_patients x T_i) x D
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        return z, {}

    def loss_kld(self, mu, log_var, *args, **kwargs):
        # v = 1.
        # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        v = self.latent_prior_noise_var
        KLD = -0.5 * torch.sum(
            1 + log_var - math.log(v) - mu.pow(2) / v - log_var.exp() / v, dim=-1
        )

        return KLD.mean(dim=0), {}


class BetaVAEgpPrior(BetaVAEgpInd):
    """
    This version correspond to a beta VAE with a prior GP in the latent space and a guidance classifier.
    """

    def __init__(
        self,
        model_config: BetaVAEgpConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        classifier=None,
    ):
        super().__init__(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
            classifier=classifier,
        )

        self.model_name = "BetaVAEgpPrior"

        # independent prior GPs in each latent dimension
        lengthscales = model_config.lengthscales
        scales = model_config.scales
        self.priorGPs = []
        for d in range(self.latent_dim):
            priorGP = PriorGP(lengthscales[d], self.latent_prior_noise_var, scales[d])
            self.priorGPs.append(priorGP)

            # add module to VAE model
            self.add_module("priorGP_" + str(d), priorGP)

            # if cuda is available, transfer model to GPU
            if torch.cuda.is_available():
                priorGP.cuda()

    # same sampling as in Independent
    # def sample(self, mu, log_var):

    def loss_kld(self, mu, log_var, times, splits, **kwargs):
        # mu: (N_patients x T_i) x D
        # log_var: (N_patients x T_i) x D

        # list of len=N_patients with tensors (T_i x L)
        mu_splitted = torch.split(mu, splits, dim=0)
        log_var_splitted = torch.split(log_var, splits, dim=0)
        times_splitted = torch.split(times, splits, dim=0)

        KLDs = torch.stack(
            [
                sum(
                    [
                        self.priorGPs[d].get_KL_post_prior(
                            times_splitted[i],
                            mu_splitted[i][:, d],
                            log_var_splitted[i][:, d].exp(),
                        )
                        for d in range(mu.shape[1])
                    ]
                )
                for i in range(len(times_splitted))
            ]
        )

        KLD = sum(KLDs)
        KLD /= mu.shape[0]  # take avaerage

        return KLD, {}

    def plot_z(
        self, z, mu, log_var, time, MS, VS, xs, figsize=(6, 3), CI=True, **kwargs
    ):
        axs = super().plot_z(z, mu, log_var, time, figsize=figsize, CI=CI, **kwargs)

        for i in range(axs.shape[0]):
            ax = axs[i]
            ax.plot(xs, MS[i, :], "-", color="C" + str(i), label="ms")

            CIadd = 1.96 * np.sqrt(np.diag(VS[i]))
            ax.fill_between(
                xs.flatten(),
                MS[i, :] - CIadd,
                MS[i, :] + CIadd,
                color="C" + str(i),
                alpha=0.1,
            )

            # ax.set_ylim(-2,2)
            ax.legend()

        return axs


class BetaVAEgpPost(BetaVAEgpPrior):
    """
    This version correspond to a beta VAE with a full posterior GP in the latent space and a guidance classifier.
    """

    def __init__(
        self,
        model_config: BetaVAEgpConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        classifier=None,
    ):
        super().__init__(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
            classifier=classifier,
        )

        self.model_name = "BetaVAEgpPost"

    def sample(self, mu, log_var, times, splits):
        # sample once from N(mu, var)
        # z: (N_patients x T_i) x D

        # mu: (N_patients x T_i) x D
        # log_var: (N_patients x T_i) x D

        # list of len=N_patients with tensors (T_i x L)
        mu_splitted = torch.split(mu, splits, dim=0)
        log_var_splitted = torch.split(log_var, splits, dim=0)
        times_splitted = torch.split(times, splits, dim=0)

        z, GPmean, Ls, Vs, ms, iPs = self.compute_posteriors(
            mu_splitted, log_var_splitted, times_splitted
        )

        # GPmean is tensor, ms is list
        keyword_dict = {
            "Vs": Vs,
            "Ls": Ls,
            "GPmean": GPmean,
            "ms": ms,
            "iPs": iPs,
            "mu_splitted": mu_splitted,
            "log_var_splitted": log_var_splitted,
            "times_splitted": times_splitted,
        }

        return z, keyword_dict

    def loss_kld(self, mu, log_var, times, splits, **kwargs):
        # mu: (N_patients x T_i) x D
        # log_var: (N_patients x T_i) x D

        # Note that we don't need mu, log_var and times
        # instead we us ms, Vs and times_splitted in kwargs already computed in the sampling step

        # list of len=N_patients with tensors (T_i x L)
        # mu_splitted = kwargs['mu_splitted']
        # log_var_splitted = kwargs['log_var_splitted']

        times_splitted = kwargs["times_splitted"]
        ms = kwargs["ms"]
        Vs = kwargs["Vs"]
        iPs = kwargs["iPs"]

        # compute personalized mean
        mu_splitted = kwargs["mu_splitted"]
        pers_means = [mu.mean(dim=0) for mu in mu_splitted]

        # mu_splitted = torch.split( mu, splits, dim=0 )
        # log_var_splitted = torch.split( log_var, splits, dim=0 )
        # times_splitted = torch.split( times, splits, dim=0 )

        KLDs = torch.stack(
            [
                sum(
                    [
                        self.priorGPs[d].get_KL_post_prior2(
                            times_splitted[i],
                            ms[i][:, d],
                            Vs[i][:, :, d],
                            pers_means[i][d],
                            iPs[i][:, :, d],
                        )
                        for d in range(ms[0].shape[1])
                    ]
                )
                for i in range(len(ms))
            ]
        )

        KLD = sum(KLDs)
        KLD /= ms[0].shape[0]  # take avaerage

        keyword_dict_for_output = {"ms": ms, "Vs": Vs, "pers_means": pers_means}

        return KLD, keyword_dict_for_output

    def plot_z(self, z, mu, log_var, time, MS, VS, xs, GPmean, CI=False, **kwargs):
        axs = super().plot_z(z, mu, log_var, time, MS, VS, xs, CI=CI, **kwargs)

        for i in range(axs.shape[0]):
            ax = axs[i]
            ax.plot(
                time, GPmean[:, i], "o", color="C" + str(i), label="GPmean", alpha=0.2
            )

            ax.legend()

        return axs

    def compute_posteriors(self, mus, log_vars, times_list):
        """
        TODO: make it much more efficient!!!!
        """

        Ls = []
        zs = []
        means = []
        Vs = []
        iPs = []

        # print('1 ',len(mus), len(self.priorGPs) )

        for i in range(len(mus)):
            varis = log_vars[i].exp()

            m_ = []
            L_ = []
            V_ = []
            z_ = []
            iP_ = []
            for d, GP in enumerate(self.priorGPs):
                m, V, L, z, iP = GP.posterior_cholesky(
                    times_list[i], mus[i][:, d], varis[:, d]
                )
                m_.append(m)
                L_.append(L)
                z_.append(z)
                V_.append(V)
                iP_.append(iP)

                # print('2 ',m.shape, L.shape, z.shape)

            # print('3 ', torch.stack(m_, dim=-1).shape, torch.stack(L_, dim=-1).shape, torch.stack(z_, dim=-1).shape)

            means.append(torch.stack(m_, dim=-1))
            Ls.append(torch.stack(L_, dim=-1))
            zs.append(torch.stack(z_, dim=-1))
            Vs.append(torch.stack(V_, dim=-1))
            iPs.append(torch.stack(iP_, dim=-1))

        return torch.cat(zs, dim=0), torch.cat(means, dim=0), Ls, Vs, means, iPs
