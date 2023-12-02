#!/usr/bin/env python
import rospy
from pytorch_lightning import Trainer
import torch
from utils import SAC
from openai_ros.task_envs.bebop2 import double_bebop2_task

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()



if __name__ == '__main__':

    rospy.init_node('SAC_bebop_train')
    
    algo = SAC('DoubleBebop2Env-v0', lr=1e-3, alpha=0.002, tau=0.1, samples_per_epoch=1000)
    algo.load_from_checkpoint("/home/huss/.ros/SAC_Models/lightning_logs/version_4/checkpoints/epoch=166-step=1336.ckpt")

    trainer = Trainer(
        gpus=num_gpus, 
        max_epochs=200000,
        log_every_n_steps=1,
        default_root_dir="SAC_Models/"
    )

    trainer.fit(algo)