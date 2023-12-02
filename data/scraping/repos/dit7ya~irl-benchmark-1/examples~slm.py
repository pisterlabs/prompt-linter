import os
# NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
os.environ['OMP_NUM_THREADS'] = '1'
from slm_lab.agent import Agent
from slm_lab.env import OpenAIEnv
from slm_lab.experiment import analysis
from slm_lab.experiment.monitor import Body, InfoSpace, enable_aeb_space
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util

from irl_benchmark.rl.slm_session import Session

spec = spec_util.get(spec_file='ppo.json', spec_name='ppo_mlp_shared_pendulum')
info_space = InfoSpace()
os.environ['PREPATH'] = util.get_prepath(spec, info_space)
os.environ['lab_mode'] = 'training'

spec['env'][0]['max_episode'] = 10

session = Session(spec, info_space)
data, agent = session.run()

print(data)
