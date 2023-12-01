import dotenv
dotenv.load_dotenv()
import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import binascii
from visualize import Joints2SMPL
from tools.visualize import parse_args

import mmcv
import numpy as np
import torch
from mogen.models import build_architecture
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mogen.utils.plot_utils import recover_from_ric
import scipy.ndimage.filters as filters

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

server_data = {}


@app.on_event('startup')
def init_data():
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + ["configs/remodiffuse/remodiffuse_t2m.py",
                                "logs/remodiffuse/remodiffuse_t2m/latest.pth",
                                "--device", "cuda"]
    args = parse_args()
    sys.argv = old_argv
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0]).to(args.device)
    model.eval()
    mean = np.load("data/datasets/human_ml3d/mean.npy")
    std = np.load("data/datasets/human_ml3d/std.npy")
    server_data["model"] = model
    server_data["mean"] = mean
    server_data["std"] = std
    server_data["j2s"] = Joints2SMPL(device=args.device)
    server_data["device"] = args.device
    return server_data


def prompt2motion(text, motion_length, model, mean, std, device):
    motion = torch.zeros(1, motion_length, 263).to(device)
    motion_mask = torch.ones(1, motion_length).to(device)
    motion_length = torch.Tensor([motion_length]).long().to(device)
    input = {
        'motion': motion,
        'motion_mask': motion_mask,
        'motion_length': motion_length,
        'motion_metas': [{'text': text}],
    }
    with torch.no_grad():
        input['inference_kwargs'] = {}
        output = model(**input)
        output = output[0]['pred_motion']
        pred_motion = output.cpu().detach()
        pred_motion = pred_motion * std + mean
    joint = recover_from_ric(pred_motion, 22).numpy()
    joint=filters.gaussian_filter1d(joint, 2.5, axis=0, mode='nearest')
    return joint


@app.get("/mld_pos/")
async def mld_pos(prompt: str):
    try:
        prompt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "translate to english without any explanation. If it's already in english, just repeat it."},
                      {"role": "user", "content": prompt}],
            timeout=10,
        )["choices"][0]["message"]["content"]
    except:
        pass
    joints = prompt2motion(prompt, 100, server_data["model"], server_data["mean"], server_data["std"],
                           server_data["device"])
    return {"positions": binascii.b2a_base64(joints.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "dtype": "float32",
            "fps": 20,
            "mode": "xyz",
            "n_frames": joints.shape[0],
            "n_joints": 22}


@app.get("/mld_angle/")
async def mld_angle(prompt: str, do_translation: bool = True):
    if do_translation:
        try:
            prompt = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system",
                           "content": "translate to english without any explanation. If it's already in english, just repeat it."},
                          {"role": "user", "content": prompt}],
                timeout=10,
            )["choices"][0]["message"]["content"]
        except:
            pass
    joints = prompt2motion(prompt, 100, server_data["model"], server_data["mean"], server_data["std"],
                           server_data["device"])
    if ((joints[:, 1, 0] > joints[:, 2, 0]) & (joints[:, 13, 0] > joints[:, 14, 0]) & (
            joints[:, 9, 1] > joints[:, 0, 1])).sum() / joints.shape[0] > 0.85:
        rotations, root_pos = server_data["j2s"](joints, step_size=1e-2, num_iters=150, optimizer="adam")
    else:
        rotations, root_pos = server_data["j2s"](joints, step_size=2e-2, num_iters=25, optimizer="lbfgs")
    return {"root_positions": binascii.b2a_base64(
        root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "rotations": binascii.b2a_base64(rotations.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "dtype": "float32",
            "fps": 20,
            "mode": "axis_angle",
            "n_frames": joints.shape[0],
            "n_joints": 24}
