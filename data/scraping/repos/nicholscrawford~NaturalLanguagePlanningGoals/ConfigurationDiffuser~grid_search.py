from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import argparse
import os
import time
import itertools
from torch.utils.data import DataLoader

from ConfigurationDiffuser.configuration_diffuser_pl import SimpleTransformerDiffuser
from Data.basic_writerdatasets_st import DiffusionDataset
from ConfigurationDiffuser.guidance_func import guidance_functions


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    
    parser = argparse.ArgumentParser(description="Grid search over sampling hyperparameters")
    parser.add_argument("--config_file", help='config yaml file',
                        default='ConfigurationDiffuser/Config/sampling_example.yaml',
                        type=str)
    args = parser.parse_args()

    assert os.path.exists(args.config_file), "Cannot find config yaml file at {}".format(args.config_file)
    cfg = OmegaConf.load(args.config_file)

    if not os.path.exists(cfg.poses_dir):
        os.makedirs(cfg.poses_dir)
    if not os.path.exists(cfg.pointclouds_dir):
        os.makedirs(cfg.pointclouds_dir)

    if len(os.listdir(os.path.join(cfg.pointclouds_dir, "1"))) < 10:
        print("Must have the show_gen_poses script running! It's needed to get the point clouds to pass into the model.")
        exit(0)

    # Prompt to confirm file deletion
    if len([file for file in os.listdir(os.path.join(cfg.pointclouds_dir,"1")) if "initial" in file]) > 0:
        confirmation = input(f"Delete all initial files in {cfg.pointclouds_dir}? (y/n): ")
        if confirmation.lower() == 'y':
            # Remove all files in the directory
            _ = [os.remove(os.path.join(os.path.join(cfg.pointclouds_dir,"1"), file)) for file in os.listdir(os.path.join(cfg.pointclouds_dir,"1")) if "initial" in file]
            print("All initial files have been deleted.")
        else:
            exit(0)

    test_dataset = DiffusionDataset(cfg.device, ds_roots=[cfg.pointclouds_dir], clear_cache=True)
    data_cfg = cfg.dataset
    test_dataloader = DataLoader(test_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)
    
    guidance_function = guidance_functions(cfg.sampling, device=cfg.device)
    # Initialize the model
    os.environ["DATETIME"] = time.strftime("%Y_%m_%d-%H:%M:%S")
    model = SimpleTransformerDiffuser.load_from_checkpoint(cfg.model_dir)
    model.poses_dir = cfg.poses_dir
    model.sampling_cfg = cfg.sampling
    if cfg.sampling.guidance_sampling:
        model.guidance_function = guidance_function.clip_guidance_function

    # Initialize the PyTorch Lightning trainer
    best_score = float('inf')
    best_hyperparameters = None
    ddim_steps_range = [20, 30]
    guidance_strength_factor_range = [10, 50, 200] #200
    backwards_steps_m_range = [5, 15, 25] #25
    backward_guidance_lr_range =[0.01, 0.05, 0.1] #0.01
    per_step_k_range =  [3, 4, 6] #6
    combinations = itertools.product(
        ddim_steps_range,
        guidance_strength_factor_range,
        backwards_steps_m_range,
        backward_guidance_lr_range,
        per_step_k_range 
    )

    trainer = pl.Trainer()

    for hyperparameters in combinations:
        ddim_steps, guidance_strength_factor, backwards_steps_m, backward_guidance_lr, per_step_k = hyperparameters

        model.sampling_cfg.ddim_steps = ddim_steps
        model.sampling_cfg.guidance_strength_factor = guidance_strength_factor
        model.sampling_cfg.backwards_steps_m = backwards_steps_m
        model.sampling_cfg.backward_guidance_lr = backward_guidance_lr
        model.sampling_cfg.per_step_k = per_step_k


        trainer.test(model, test_dataloader)
        score = model.guidance_alignment.mean()
        if score < best_score:
            best_score = score
            best_hyperparameters = hyperparameters
        print("Hyperparameters:", hyperparameters)
        print("Score:", score.item())

    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Score:", best_score)
    print("Best Possible Score: ~70.18")