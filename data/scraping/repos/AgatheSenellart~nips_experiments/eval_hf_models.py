"""

Script to compute more evaluation metrics on the MMNIST experiments

"""

import argparse
import json
import os
import tempfile

import numpy as np
import torch
from config2 import *
from huggingface_hub import CommitOperationAdd, HfApi
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, FIDEvaluator
from multivae.models.auto_model import AutoConfig, AutoModel
from multivae.models.base.base_model import BaseEncoder, ModelOutput
from torchvision.utils import make_grid
from PIL import Image
from multivae.metrics import Visualization, VisualizationConfig
from config2 import *

##############################################################################


train_set = MMNISTDataset(data_path=data_path, split="train", download = True)
test_set = MMNISTDataset(data_path=data_path, split="test", download = True)

# Get config_files
parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
info['model_name'] = args.model_name
args = argparse.Namespace(**info)

missing_ratio = "".join(str(args.missing_ratio).split("."))
incomplete = "i" if args.keep_incomplete else "c"

# Load the pretrained model from hf hub
hf_repo = f"asenella/mmnist_{args.model_name}{config_name}_seed_{args.seed}_ratio_{missing_ratio}_{incomplete}"
model = AutoModel.load_from_hf_hub(hf_repo, allow_pickle=True)

if torch.cuda.is_available():   
    model = model.cuda()
    model.device = "cuda"
else :
    model.cpu()
    model.device ='cpu'


id = f'{args.model_name}_{incomplete}_{missing_ratio}_{args.seed}'


output_dir = f'{output_path}/validate_mmnist/{args.model_name}/incomplete_{incomplete}/missing_ratio_{missing_ratio}/seed_{args.seed}'

# Recompute the cross-coherences and joint coherence from prior and FID if necessary
config = CoherenceEvaluatorConfig(batch_size=512)

CoherenceEvaluator(
    model=model,
    test_dataset=test_set,
    classifiers=load_mmnist_classifiers(device=model.device),
    output=output_dir,
    eval_config=config,
).eval()

# Visualize unconditional samples and conditional samples too
vis_config = VisualizationConfig(n_samples=8, n_data_cond=10)
vis_module = Visualization(model, test_set,eval_config=vis_config,output = output_dir)
vis_module.eval()

# And some conditional samples too
for i in range(2,5):
    subset = modalities[1:1+i]
    vis_module.conditional_samples_subset(subset)

vis_module.finish()

# Compute confiditional FIDs
fid_config = FIDEvaluatorConfig(batch_size=128,
                                inception_weights_path=os.path.join(data_path,'pt_inception-2015-12-05-6726825d.pth'))


FIDEvaluator(
        model, test_set, output=output_dir, eval_config=fid_config
    ).compute_all_conditional_fids(gen_mod="m0")


#### Compute joint coherence for other samplers

# From MAF sampler
from multivae.samplers import MAFSampler, MAFSamplerConfig
from pythae.trainers import BaseTrainerConfig

training_config = BaseTrainerConfig(per_device_train_batch_size=256, num_epochs=500, learning_rate=1e-3)
sampler_config = MAFSamplerConfig()
maf_sampler = MAFSampler(model)
train_data, eval_data = random_split(
    train_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)
maf_sampler.fit(train_data = train_data, eval_data = eval_data,training_config=training_config)


# From GMM sampler
from multivae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig
from pythae.trainers import BaseTrainerConfig


gmm_sampler = GaussianMixtureSampler(model)
gmm_sampler.fit(train_set)

samplers = [maf_sampler, gmm_sampler]

for sampler in samplers:
    config = CoherenceEvaluatorConfig(batch_size=128)
    module_eval = CoherenceEvaluator(model,load_mmnist_classifiers(),test_set, eval_config=config,sampler=sampler)
    module_eval.joint_coherence()
    module_eval.log_to_wandb()
    module_eval.finish()


    config = FIDEvaluatorConfig(batch_size=128, 
                                inception_weights_path=os.path.join(data_path,'pt_inception-2015-12-05-6726825d.pth'))
    module_eval = FIDEvaluator(model,test_set,eval_config=config, sampler=sampler)
    module_eval.eval()
    module_eval.finish()
    
    
##### Compute k-means clustering

from multivae.metrics import Clustering, ClusteringConfig

c_config = ClusteringConfig(number_of_runs=10,
    num_samples_for_fit=None, 
   )
c = Clustering(model, test_set, train_set,eval_config=c_config)
c.eval()
c.finish()

##### Compute joint likelihood
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig

lik_config = LikelihoodsEvaluatorConfig(
    batch_size=128,
   
    num_samples=1000,
    batch_size_k=250,
)

lik_module = LikelihoodsEvaluator(model,
                                  test_set,
                                  output= output_dir,
                                  eval_config=lik_config,
                                  )
lik_module.eval()
lik_module.finish()
