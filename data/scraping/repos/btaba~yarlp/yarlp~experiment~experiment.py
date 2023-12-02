import os
import gym
import sys
import json
import copy
import click
import subprocess
import pandas as pd
import numpy as np

from jsonschema import validate
from itertools import product
from yarlp.experiment import plotting
from concurrent.futures import ProcessPoolExecutor
from yarlp.experiment.experiment_schema import schema
from yarlp.experiment.job import Job
from yarlp.utils import experiment_utils


class Experiment(object):
    def __init__(self, video=False):
        """
        Params
        ----------
        video (bool): False disables video recording. otherwise
            we us the defaul openai gym behavior of taking videos on
            every cube up until 1000, and then for every 1000 episodes.
        """
        self.video = video
        self.reload_exp = False

    @classmethod
    def from_json_spec(cls, json_spec_filename, log_dir=None, reload_exp=False,
                       *args, **kwargs):
        """
        Reads in json_spec_filen of experiment, validates the experiment spec,
        creates a spec for each combination of agent/env/grid-search-params
        and creates the experiment directory
        Params
        ----------
        json_spec_filename (str): the file path of the json spec file for the
            complete experiment
        reload_exp (bool): if True, try to reload experiment from a directory
        """
        cls = cls(*args, **kwargs)
        cls.reload_exp = reload_exp
        assert json_spec_filename is not None
        assert os.path.exists(json_spec_filename) and\
            os.path.isfile(json_spec_filename),\
            "spec-filename does not exist"

        spec_file_handle = open(json_spec_filename, 'r')
        _raw_spec = json.load(spec_file_handle)

        # validate the json spec using jsonschema
        validate(_raw_spec, schema)

        # create a json spec for each env-agent-repeat
        # _spec_list = cls._spec_product(_raw_spec)
        _spec_list = _raw_spec['runs']

        cls._validate_agent_names(_spec_list)
        _spec_list = cls._add_validated_agent_repeats(_spec_list)

        # if the agent params are lists, then we need to split them up further
        # since these will be the cross-validation (cv) folds
        cls._spec_list = cls._expand_agent_grid(_spec_list)

        # create log directory and save the full spec to the directory
        if not log_dir:
            cls._experiment_dir = cls._create_log_dir(
                json_spec_filename)
        else:
            cls._experiment_dir = log_dir
        experiment_utils._save_spec_to_dir(_raw_spec, cls._experiment_dir)
        return cls

    def run(self, parallel=True, n_jobs=None):
        if not parallel:
            # GUI operations don't play nice with parallel execution
            for j in self._jobs:
                j()
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                ex.map(self.run_job, self._spec_list)

    def run_job(self, s):
        env = os.environ.copy()
        python_path = sys.executable
        command = ('{} -m yarlp.experiment.job --log-dir {}'
                   ' --video {} --spec \'{}\' --reload-exp \'{}\'').format(
            python_path, self._experiment_dir, self.video, json.dumps(s),
            self.reload_exp)
        p = subprocess.Popen(command, env=env, shell=True)
        out, err = p.communicate()
        return out, err

    @property
    def _jobs(self):
        for s in self._spec_list:
            yield Job(s, self._experiment_dir, self.video, self.reload_exp)

    @property
    def spec_list(self):
        return self._spec_list

    def _add_validated_agent_repeats(self, spec_list):
        """
        Validates the environment names and adds a json spec
        for each repeat with a 'run_name'
        """
        env_list = [x.id for x in gym.envs.registry.all()]

        repeated_spec_list = []
        for s in spec_list:
            env_name = s['env']['name']
            assert env_name in env_list,\
                "{} is not an available environment name.".format(env_name)

            for r in set(s['agent']['seeds']):
                s_copy = copy.deepcopy(s)
                run_name = '{}_{}_run{}'.format(
                    s['env']['name'], s['agent']['type'], r)
                s_copy['run_name'] = run_name
                s_copy['seed'] = r
                s_copy['agent']['seeds'] = [r]
                repeated_spec_list.append(s_copy)

        return repeated_spec_list

    def _expand_agent_grid(self, spec_list):
        """
        If the agent params are lists, we need to expand them
        into one run for each parameter grid...effectively doing a grid search
        over a list of parameters
        """

        grid_search = []

        for scount, s in enumerate(spec_list):

            # expand the grid of params
            params = s['agent'].get('params', {})
            singleton_params = {
                k: v for k, v in params.items() if not isinstance(v, list) or
                k.endswith('schedule')}
            grid_params = {
                k: v for k, v in params.items() if isinstance(v, list) and
                not k.endswith('schedule')}
            grid_params = [
                dict(zip(grid_params.keys(), x))
                for x in product(*grid_params.values())]
            grid_params = [{**g, **singleton_params} for g in grid_params]

            # add a spec with each agent param in the grid
            count = 0
            for g in grid_params:
                new_s = copy.deepcopy(s)
                new_s['agent']['params'] = g
                new_s['param_run'] = count
                param_name = '_param{}_spec{}'.format(count, scount)
                new_s['run_name'] = new_s['run_name'] + param_name
                grid_search.append(new_s)
                count += 1

        return grid_search

    def _validate_agent_names(self, spec):
        agent_set = set()
        cls_dict = experiment_utils._get_agent_cls_dict()
        for s in spec:
            agent_name = s['agent']['type']
            assert agent_name in cls_dict,\
                "{} is not an implemented agent. Select one of {}".format(
                    agent_name, cls_dict.keys())

            agent_set.add(agent_name)

    @staticmethod
    def _get_experiment_dir(experiment_name):
        home = os.path.expanduser('~')
        experiment_dir = os.path.join(
            home, 'yarlp_experiments', experiment_name)
        return experiment_dir

    def _create_log_dir(self, spec_filename):
        base_filename = os.path.basename(spec_filename)
        experiment_name = base_filename.split('.')[0]

        experiment_dir = Experiment._get_experiment_dir(
            experiment_name)

        return experiment_utils._create_log_directory(
            experiment_name, experiment_dir)


def _merge_stats(experiment_dir):
    """
    Loop through all experiments, and write all the stats
    back to the base repository
    """
    statspath = os.path.join(experiment_dir, 'stats')
    if not os.path.exists(statspath):
        os.makedirs(statspath)
    agg_stats_file = os.path.join(statspath, 'merged_stats.tsv')

    stats_list = []
    for d in os.listdir(experiment_dir):
        base_path = os.path.join(experiment_dir, d)
        if not os.path.isdir(base_path) or d == 'stats':
            continue
        spec = open(os.path.join(base_path, 'spec.json'), 'r')
        spec = json.load(spec)

        f = os.path.join(base_path, 'stats.json.txt')
        with open(f, 'r') as f:
            stats = list(map(json.loads, f.readlines()))
        stats = pd.DataFrame(stats)

        spec = spec['runs']
        stats['param_run'] = spec['param_run']
        stats['run_name'] = spec['run_name']
        stats['agent'] = spec['agent']['type']
        stats['env'] = spec['env']['name']
        stats['agent_params'] = str(spec['agent']['params'])
        stats['seed'] = spec['seed']
        stats_list.append(stats)

    assert len(stats_list) > 0, "No stats were found."
    stats = stats_list[0]
    for s in stats_list[1:]:
        stats = stats.append(s)

    with open(agg_stats_file, 'w') as f:
        stats.to_csv(f, index=False, header=True, sep='\t')

    return stats


def _merge_benchmark_stats(experiment_dir):
    """
    Merge stats from OpenAI benchmarks experiments
    """
    statspath = os.path.join(experiment_dir, 'stats')
    if not os.path.exists(statspath):
        os.makedirs(statspath)
    agg_stats_file = os.path.join(statspath, 'merged_stats.tsv')

    stats_list = []
    for d in os.listdir(experiment_dir):
        base_path = os.path.join(experiment_dir, d)
        if not os.path.isdir(base_path) or d == 'stats':
            continue

        stats = pd.read_csv(os.path.join(base_path, 'progress.csv'))

        f = os.path.join(base_path, '0.monitor.csv')
        with open(f, 'r') as f:
            spec = f.readline()
            spec = json.loads(spec[1:])

        stats['env'] = spec['env_id']
        stats['run_name'] = d

        stats_list.append(stats)

    assert len(stats_list) > 0, "No stats were found."
    stats = stats_list[0]
    for s in stats_list[1:]:
        stats = stats.append(s)

    with open(agg_stats_file, 'w') as f:
        stats.to_csv(f, index=False, header=True, sep='\t')

    return stats


def generate_plots(yarlp_dir, by_field='env'):
    ts = _merge_stats(yarlp_dir)
    ts['run_name'] = ts['run_name'].apply(
        lambda x: '_'.join(x.split('_')[-2:]))
    for rn in ts['run_name'].unique():
        ts.loc[ts['run_name'] == rn, 'Iteration'] = np.arange(
            ts[ts['run_name'] == rn].shape[0])
    for f in ts[by_field].unique():
        ts['name'] = 'yarlp'
        yarlp = ts[ts[by_field] == f]
        fig = plotting.make_plots(yarlp, f, 'run_name', 'param_run')
        fig.savefig(
            os.path.join(
                yarlp_dir,
                '{}.png'.format(f)))

        fig = plotting.make_plots(yarlp, f, 'run_name', 'run_name')
        fig.savefig(
            os.path.join(
                yarlp_dir,
                '{}_all_runs.png'.format(f)))


def generate_plots_benchmark_vs_yarlp(yarlp_dir, benchmark_dir):
    training_stats = _merge_stats(yarlp_dir)
    benchmark_stats = _merge_benchmark_stats(benchmark_dir)
    for env in training_stats.env.unique():
        benchmark = benchmark_stats[benchmark_stats.env == env].rename(
            columns={'EpRewMean': 'Smoothed_total_reward',
                     'TimestepsSoFar': 'timesteps_so_far',
                     'TimeElapsed': 'time_elapsed',
                     'env': 'env_id'})
        benchmark['name'] = 'benchmark'
        benchmark['episode'] = 0
        for run_name in benchmark.run_name.unique():
            benchmark.loc[benchmark.run_name == run_name, 'Iteration'] = list(
                range(benchmark[benchmark.run_name == run_name].shape[0]))

        training_stats['name'] = 'yarlp'
        yarlp = training_stats[training_stats.env == env]

        merged_data = pd.concat([benchmark, yarlp])

        fig = plotting.make_plots(merged_data, env)

        fig.savefig(
            os.path.join(
                yarlp_dir,
                '{}.png'.format(env)))


def get_benchmarks(benchmark_name):
    from yarlp.experiment.benchmarks import _BENCHMARKS
    benchmark_dict = dict(
        map(lambda x: (x[1]['name'], x[0]), enumerate(_BENCHMARKS)))
    assert benchmark_name in benchmark_dict
    benchmark_idx = benchmark_dict[benchmark_name]
    benchmark = _BENCHMARKS[benchmark_idx]
    return benchmark


@click.group()
def cli():
    pass


@click.command()
@click.option('--agent', default='TRPOAgent')
def run_mujoco1m_benchmark(agent):
    SEEDS = list(range(652, 752))

    benchmark_name = 'Mujoco1M'

    # Make a master log directory
    experiment_dir = Experiment._get_experiment_dir(
        benchmark_name)
    base_log_path = experiment_utils._create_log_directory(
        benchmark_name, experiment_dir)
    benchmark = get_benchmarks(benchmark_name)
    # write the json config for this baseline
    j = []
    for t in benchmark['tasks']:
        d = {
            "env": {
                "name": t['env_id'],
                "normalize_obs": True
            },
            "agent": {
                "type": agent,
                "seeds": SEEDS[:t['trials']],
                "training_params": {
                    "max_timesteps": t['num_timesteps']
                }
            }
        }
        j.append(d)

    j = {"runs": j}
    spec_file = os.path.join(base_log_path, 'spec.json')
    json.dump(j, open(spec_file, 'w'))

    # run the experiment
    e = Experiment.from_json_spec(
        spec_file, log_dir=base_log_path)
    e.run()


@click.command()
@click.option('--n-jobs', default=1)
def run_atari10m_ddqn_benchmark(n_jobs):
    agent = 'DDQNAgent'
    SEEDS = list(range(652, 752))

    benchmark_name = 'Atari10M'

    # Make a master log directory
    experiment_dir = Experiment._get_experiment_dir(
        benchmark_name)
    base_log_path = experiment_utils._create_log_directory(
        benchmark_name, experiment_dir)
    benchmark = get_benchmarks(benchmark_name)
    # write the json config for this baseline
    j = []
    print(benchmark['tasks'])
    for t in benchmark['tasks']:
        d = {
            "env": {
                "name": t['env_id'],
                "is_atari": True
            },
            "agent": {
                "type": agent,
                "seeds": [SEEDS[0]],
                "training_params": {},
                "params": {
                    "discount_factor": 0.99,
                    "learning_start_timestep": 10000,
                    "buffer_size": 1000000,
                    "train_freq": 4,
                    "policy_learning_rate": 0.0001,
                    "max_timesteps": t['num_timesteps'],
                    "target_network_update_freq": 10000,
                    "save_freq": 50000,
                    "prioritized_replay": False,
                    "double_q": True,
                    "policy_network_params": {"dueling": True},
                    "learning_rate_schedule": [
                        [0, 1e-4],
                        [1e6, 1e-4],
                        [5e6, 5e-5]
                    ],
                    "exploration_schedule": [
                        [0, 1.0],
                        [1e6, 0.1],
                        [5e6, 0.01]
                    ]
                }
            }
        }
        j.append(d)

    j = {"runs": j}
    spec_file = os.path.join(base_log_path, 'spec.json')
    json.dump(j, open(spec_file, 'w'))

    # run the experiment
    e = Experiment.from_json_spec(
        spec_file, log_dir=base_log_path, video=True)
    if n_jobs > 1:
        e.run(parallel=True, n_jobs=n_jobs)
    else:
        e.run(parallel=False)


@click.command()
def run_atari10m_a2c_benchmark():
    agent = 'A2CAgent'
    SEEDS = list(range(652, 752))

    benchmark_name = 'Atari10M'

    # Make a master log directory
    experiment_dir = Experiment._get_experiment_dir(
        benchmark_name)
    base_log_path = experiment_utils._create_log_directory(
        benchmark_name, experiment_dir)
    benchmark = get_benchmarks(benchmark_name)
    # write the json config for this baseline
    j = []
    print(benchmark['tasks'])
    for t in benchmark['tasks']:
        d = {
            "env": {
                "name": t['env_id'],
                "is_atari": True,
                "num_envs": 16,
                "is_parallel": True
            },
            "agent": {
                "type": agent,
                "seeds": [SEEDS[0]],
                "training_params": {},
                "params": {
                    "discount_factor": 0.99,
                    "max_timesteps": t['num_timesteps'],
                    "save_freq": 50000,
                    "policy_learning_rate_schedule": [
                        [0, 7e-4],
                        [20e6, 1e-12]
                    ],
                    "policy_network_params": {
                        "final_dense_weights_initializer": 0.1
                    },
                    "value_network_params": {
                        "final_dense_weights_initializer": 1.0
                    },
                    "grad_norm_clipping": 1,
                    "entropy_weight": 0.01,
                    "n_steps": 5
                }
            }
        }
        j.append(d)

    j = {"runs": j}
    spec_file = os.path.join(base_log_path, 'spec.json')
    json.dump(j, open(spec_file, 'w'))

    # run the experiment
    e = Experiment.from_json_spec(
        spec_file, log_dir=base_log_path, video=True)
    e.run(parallel=False)


@click.command()
@click.option('--spec-file',
              default='./experiment_configs/reinforce_experiment.json',
              help=('Path to json file spec if continue=False'
                    ', else path to experiment'))
@click.option('--video', default=False, type=bool,
              help='Whether to record video or not')
@click.option('--parallel', default=True, type=bool,
              help='Whether to run in parallel or not')
@click.option('--n-jobs', default=1)
@click.option('--reload-exp', default=False, type=bool,
              help='Whether to restart from an experiment dir')
def run_experiment(spec_file, video, parallel, n_jobs, reload_exp):
    if reload_exp:
        log_dir = os.path.dirname(spec_file)
    else:
        log_dir = None
    e = Experiment.from_json_spec(
        spec_file, log_dir=log_dir, video=video,
        reload_exp=reload_exp)
    e.run(parallel=parallel, n_jobs=n_jobs)


@click.command()
@click.option(
    '--upload-dir',
    help='Path of openai gym session to upload')
def upload_to_openai(upload_dir):
    gym.scoreboard.api_key = os.environ.get('OPENAI_GYM_API_KEY', None)
    gym.upload(upload_dir)


@click.command()
@click.argument('yarlp-dir')
@click.argument('openai-benchmark-dir')
def compare_benchmark(yarlp_dir, openai_benchmark_dir):
    generate_plots_benchmark_vs_yarlp(yarlp_dir, openai_benchmark_dir)


@click.command()
@click.argument('directory')
@click.option('--by-field', default='env')
def make_plots(directory, by_field):
    generate_plots(directory, by_field)


cli.add_command(run_mujoco1m_benchmark)
cli.add_command(run_atari10m_ddqn_benchmark)
cli.add_command(run_atari10m_a2c_benchmark)
cli.add_command(run_experiment)
cli.add_command(upload_to_openai)
cli.add_command(compare_benchmark)
cli.add_command(make_plots)


if __name__ == '__main__':
    cli()
