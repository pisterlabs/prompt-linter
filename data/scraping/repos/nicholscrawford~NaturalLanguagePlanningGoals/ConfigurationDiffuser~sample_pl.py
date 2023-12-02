from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import argparse
import os
import time
from torch.utils.data import DataLoader

from ConfigurationDiffuser.configuration_diffuser_pl import SimpleTransformerDiffuser
from Data.basic_writerdatasets_st import DiffusionDataset
from ConfigurationDiffuser.guidance_func import guidance_functions


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    
    parser = argparse.ArgumentParser(description="Run a simple model")
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
        model.guidance_function = guidance_function.clip_guidance_function_logits
        # model.guidance_function = guidance_function.clip_guidance_function
        # model.guidance_function = guidance_function.away_from_each_other_guidance_function

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer()
    loss = trainer.test(model, test_dataloader)
    # print(model.guidance_alignment)
