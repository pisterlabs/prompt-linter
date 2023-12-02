import cv2
import gym
import numpy as np
import torch
import torchvision.transforms as T

from drl.deepq.game import Action
from drl.openai.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, ClipRewardEnv
from drl.openai.game_openai import OpenAIGame


class Breakout(OpenAIGame):
  actions = [Action('noop', 0, 0),
             Action('fire', 1, 1),
             Action('right', 2, 2),
             Action('left', 3, 3)]

  def __init__(self, x: int, y: int, t: int, skip=4,
               store_frames_as: torch.dtype = torch.float, force_scale_01: bool = False):
    self.transform = T.Compose([T.ToPILImage(), T.Resize((y, x)), T.Grayscale(), T.ToTensor()])
    self.scale = 255 if not store_frames_as.is_floating_point else None
    scale2 = 255 if not store_frames_as.is_floating_point and force_scale_01 else None
    super().__init__('BreakoutNoFrameskip-v4', t, store_frames_as, scale2)
    self.env = MaxAndSkipEnv(self.env, skip=skip)

  @property
  def name(self) -> str:
    return 'breakout'

  def _get_frame(self, env_state):
    raw = env_state[..., :3]
    image = self.transform(raw)
    if self.scale is not None:
      image = image * self.scale
    return image.squeeze(0)
