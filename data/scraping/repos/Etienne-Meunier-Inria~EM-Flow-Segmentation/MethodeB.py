from Models.CoherenceNet import CoherenceNet
import torch
from sklearn.linear_model import HuberRegressor
import numpy as np
from ipdb import set_trace
import cv2
from argparse import ArgumentParser


class MethodeB(CoherenceNet) :
    """
    Model using as criterion the coherence of the optical flow in segmented regions.
    """
    def __init__(self, theta_method,**kwargs) :
        super().__init__(**kwargs) # Build Coherence Net
        self.theta_method = theta_method
        self.hparams.update({'theta_method':theta_method})

    def ComputeTheta(self, batch):
        """
        General Method to compute theta

        Params :
            batch : dictionnary with at least 2 keys
                'Pred' (b, L, I, J) : Mask proba predictions
                'Flow' ( b, 2, I, J) : Flow Map
        Returns :
            Theta : parameters set for each layers and sample : (b, L, ft, 2)
        """
        if self.theta_method == 'OLS' :
            return self.ComputeThetaOLS(batch, self.XoT)
        if self.theta_method == 'Optim' :
            return self.ComputeThetaOptim(batch, self.XoT, self.vdist)
        else :
            raise Exception(f'Method {self.theta_method} is not defined for theta computation')

    @staticmethod
    def ComputeThetaOLS(batch, XoT) :
        """
        Compute Theta using OLS

        Params :
            batch : dictionnary with at least 2 keys
                'Pred' (b, L, I, J) : Mask proba predictions
                'Flow' ( b, 2, I, J) : Flow Map
            XoT : Features for regression (I*J, ft) depending on parametric model
        Returns :
            Theta : parameters set for each layers and sample : (b, L, ft, 2)
        """
        w_masks, flows  = batch['Pred'], batch['Flow']
        b, L, I, J =  w_masks.shape
        _, ft = XoT.shape
        K = batch['Flow'].shape[1]
        flat_flows = flows.view(b, K, I*J).permute(0,2,1)# [b, 2, i*j]
        w_masks = w_masks.reshape(b, L, I*J).unsqueeze(-1) # [b, L, i*j, 1]
        w_masks_flat = w_masks.view(b*L, I*J, 1) # [b*l, i*j, 1]
        Y = flat_flows[:,None,:].expand(-1, L,-1,-1).reshape(b*L,I*J, K) # [b*l, i*j,  2]
        X = XoT.unsqueeze(0).expand(b*L, -1, -1) # [b*l, i*j, 3]
        Xw = (X * w_masks_flat.expand(-1,-1, ft)) # [b*l, i*j, ft]
        XwT = Xw.permute(0,2,1) # [b*l, ft, i*j]

        try :
            R = torch.bmm(XwT, X) # [b*l, ft, ft]
            Ri = torch.linalg.inv(R) # [b*l, ft, ft]
            M = torch.bmm(XwT, Y) # [b*l, ft, 2]  <--
            Theta = torch.bmm(Ri, M).view(b, L, ft, K) # [b, l, ft, 2]

        except Exception as e:
            Theta = torch.zeros((b, L, ft, K), device=flat_flows.device, requires_grad=True)
            print(f'Inversion Error : {e} batch : {batch["FlowPath"]}')
        if torch.isnan(Theta).sum() > 0 :
            print('Nan in Theta')
        return Theta.nan_to_num(0) # Avoid nan in theta estimation

    @staticmethod
    def ComputeThetaOptim(batch, XoT, vdist) :
        """
        Compute Theta using optimisation and the Coherence Loss

        Params :
            batch : dictionnary with at least 2 keys
                'Pred' (b, L, I, J) : Mask proba predictions
                'Flow' ( b, 2, I, J) : Flow Map
        Returns :
            Theta : parameters set for each layers and sample : (b, L, ft, 2)
        """
        b, L, I, J = list(batch['Pred'].shape)
        theta = MethodeB.ComputeThetaOLS(batch, XoT).detach() # Init with OLS
        theta.requires_grad_(True)
        lbgfs = torch.optim.LBFGS([theta], line_search_fn='strong_wolfe')
        def closure() :
            lbgfs.zero_grad()
            loss = MethodeB.CoherenceLoss(theta, batch['Pred'].detach(), batch['Flow'], XoT, vdist).sum()
            loss.backward()
            return loss
        lbgfs.step(closure)
        theta = theta.detach() # Our theta is not supposed to have gradients after LBFGS
        return theta

    @staticmethod
    def add_specific_args(parent_parser):
        parent_parser = MethodeB.__bases__[0].add_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--theta_method', help='Method used to compute theta', type=str, choices=['OLS','Optim'], default='Optim')
        return parser
