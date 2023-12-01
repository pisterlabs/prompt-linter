"""Classifier-full guidance

'Classifier-full' is a term we use for the type of guidance where we assume access to an (approx.) likelihood function
p(y | x_t), forall t = 0, ..., T.

That is, we need access to a classifier for all noise levels.
"""
import torch as th
from torch import nn
from src.guidance.base import Guidance


class ClassifierFullGuidance(Guidance):
    def __init__(self, classifier: nn.Module, loss: nn.Module, lambda_: float = 1.0):
        """
        @param classifier: Classifier model p(y|x_t , t)
        @param loss: Corresponding loss for the classifier
        @param lambda_: Magnitude of the gradient
        """
        super(ClassifierFullGuidance, self).__init__(lambda_=lambda_)
        self.classifier = classifier
        self.loss = loss

    def grad(self, x_t, t, y):
        x_t.requires_grad = True
        loss = self.loss(self.classifier(x_t, t), y)
        return self.lambda_ * th.autograd.grad(loss, x_t, retain_graph=True)[0]
