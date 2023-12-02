from guidance_computer import guidance
from ..config import atlas_config

guidance.set(atlas_config.rocket)
guidance.launch([
    forward = atlas_config.forward,
    reverse = None,
    yaw = atlas_config.yaw,
    pitch = atlas_config.pitch
])