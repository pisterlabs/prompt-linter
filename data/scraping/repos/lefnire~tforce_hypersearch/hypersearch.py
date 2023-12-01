"""
Uses Bayesian Optimization to search hyperparamter space for best-performing TensorForce model.

Each hyper is specified as `key: {type, vals, requires, hook}`.
- type: (int|bounded|bool). bool is True|False param, bounded is a float between min & max, int is "choose one"
    eg 'activation' one of (tanh|elu|selu|..)`)
- vals: the vals this hyper can take on. If type(vals) is primitive, hard-coded at this value. If type is list, then
    (a) min/max specified inside (for bounded); (b) all possible options (for 'int'). If type is dict, then the keys
    are used in the searching (eg, look at the network hyper) and the values are used as the configuration.
- guess: initial guess (supplied by human) to explore
- pre/post/hydrate (hooks): transform this hyper before plugging it into Configuration(). Eg, we'd use type='bounded'
    for batch size since we want to range from min to max (instead of listing all possible values); but we'd cast it
    to an int inside hook before using it. (Actually we clamp it to blocks of 8, as you'll see).

The special sauce is specifying hypers as dot-separated keys, like `memory.type`. This allows us to easily
mix-and-match even within a config-dict. Eg, you can try different combos of hypers within `memory{}` w/o having to
specify the whole block combo (`memory=({this1,that1}, {this2,that2})`). To use this properly, make sure to specify
a `requires` field where necessary.
"""
import argparse, json, math, time, pdb
from pprint import pprint
from box import Box
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy.sql import text
from tensorforce.agents import agents as agents_dict
from tensorforce.core.networks.layer import Dense
from tensorforce.core.networks.network import LayeredNetwork
from tensorforce.execution import Runner
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from tensorforce.contrib.openai_gym import OpenAIGym

import data


def build_net_spec(hypers, baseline=False):
    """Builds a net_spec from some specifications like width, depth, etc"""
    net = Box(hypers['net'])

    dense = {'type': 'dense', 'activation': net.activation, 'l2_regularization': net.l2, 'l1_regularization': net.l1}
    dropout = {'type': 'dropout', 'rate': net.dropout}
    conv2d = {'type': 'conv2d', 'bias': True, 'l2_regularization': net.l2,
              'l1_regularization': net.l1}  # TODO bias as hyper?
    lstm = {'type': 'internal_lstm', 'dropout': net.dropout}

    # For LSTM baselines, we don't want the LSTM cells nor the pre-layer
    lstm_baseline = net.type == 'lstm' and baseline

    arr = []
    if net.dropout:
        arr.append({**dropout})

    # Pre-layer
    if 'pre_depth' in net and not lstm_baseline:
        for i in range(net.pre_depth):
            size = int(net.width / (net.pre_depth - i + 1)) if net.funnel else net.width
            arr.append({'size': size, **dense})
            if net.dropout: arr.append({**dropout})

    # Mid-layer
    if not lstm_baseline:
        for i in range(net.depth):
            if net.type == 'conv2d':
                size = max([32, int(net.width / 4)])
                # FIXME most convs have their first layer smaller... right? just the first, or what?
                if i == 0:
                    size = int(size / 2)
                arr.append({'size': size, 'window': net.window, 'stride': net.stride, **conv2d})
            else:
                # arr.append({'size': net.width, 'return_final_state': (i == net.depth-1), **lstm})
                arr.append({'size': net.width, **lstm})
        if net.type == 'conv2d':
            arr.append({'type': 'flatten'})

    # Dense
    for i in range(net.depth):
        size = int(net.width / (i + 2)) if net.funnel else net.width
        arr.append({'size': size, **dense})
        if net.dropout: arr.append({**dropout})

    return arr


def bins_of_8(x):
    """Converts an int/float to 8-bin chunks (32, 64, etc)"""
    return int(x // 8) * 8


hypers = {}
hypers['agent'] = {}
hypers['batch_agent'] = {
    'batch_size': {
        'type': 'bounded',
        'vals': [8, 2048],
        'guess': 1024,
        'pre': bins_of_8
    },
    # 'keep_last_timestep': True
}
hypers['model'] = {
    'optimizer.type': {
        'type': 'int',
        'vals': ['adam', 'nadam'],
    },
    'optimizer.learning_rate': {
        'type': 'bounded',
        'vals': [0., 1.],
        'guess': .01,
    },
    'optimization_steps': {
        'type': 'bounded',
        'vals': [1, 20],
        'guess': 10,
        'pre': int
    },
    'discount': {
        'type': 'bounded',
        'vals': [.95, .99],
        'guess': .96
    },
    # TODO variable_noise
}
hypers['distribution_model'] = {
    'entropy_regularization': {
        'type': 'bounded',
        'vals': [0., 1.],
        'guess': .2
    }
    # distributions_spec (gaussian, beta, etc). Pretty sure meant to handle under-the-hood, investigate
}
hypers['pg_model'] = {
    'baseline_mode': {
        'type': 'bool',
        'guess': True,
        'hydrate': lambda x, flat: {
            False: {'baseline_mode': None},
            True: {
                'baseline': {'type': 'custom'},
                'baseline_mode': 'states',
                'baseline_optimizer': {
                    'type': 'multi_step',
                    'num_steps': flat['optimization_steps'],
                    'optimizer': {
                        'type': flat['step_optimizer.type'],
                        'learning_rate': flat['step_optimizer.learning_rate']
                    }
                },
            }
        }[x]
    },
    'gae_lambda': {
        'type': 'bounded',
        'vals': [.85, .99],
        'guess': .97,
        'post': lambda x, others: x if others['baseline_mode'] and x > .9 else None
    },
}
hypers['pg_prob_ration_model'] = {
    # I don't know what values to use besides the defaults, just guessing. Look into
    'likelihood_ratio_clipping': {
        'type': 'bounded',
        'vals': [0., 1.],
        'guess': .2,
        'post': lambda x, others: None if x < .05 else x
    }
}

hypers['ppo_agent'] = {  # vpg_agent, trpo_agent
    **hypers['agent'],
    **hypers['batch_agent'],
    **hypers['model'],
    **hypers['distribution_model'],
    **hypers['pg_model'],
    **hypers['pg_prob_ration_model']

}
hypers['ppo_agent']['step_optimizer.learning_rate'] = hypers['ppo_agent'].pop('optimizer.learning_rate')
hypers['ppo_agent']['step_optimizer.type'] = hypers['ppo_agent'].pop('optimizer.type')

hypers['custom'] = {
    'net.depth': {
        'type': 'bounded',
        'vals': [1, 5],
        'guess': 2,
        'pre': int
    },
    'net.width': {
        'type': 'bounded',
        'vals': [32, 768],
        'guess': 384,
        'pre': bins_of_8
    },
    'net.funnel': {
        'type': 'bool',
        'guess': True
    },
    # 'net.type': {'type': 'int', 'vals': ['lstm', 'conv2d']},  # gets set from args.net_type
    'net.activation': {
        'type': 'int',
        'vals': ['tanh', 'elu', 'relu', 'selu'],
        'guess': 'tanh'
    },
    'net.dropout': {
        'type': 'bounded',
        'vals': [0., .5],
        'guess': .2,
        'pre': lambda x: None if x < .1 else x
    },
    'net.l2': {
        'type': 'bounded',
        'vals': [0., .1],
        'guess': .04
    },
    'net.l1': {
        'type': 'bounded',
        'vals': [0., .1],
        'guess': .02
    },
}

hypers['lstm'] = {
    'net.pre_depth': {
        'type': 'bounded',
        'vals': [0, 3],
        'guess': 1,
        'pre': int,
    },
}
hypers['conv2d'] = {
    # 'net.bias': {'type': 'bool'},
    'net.window': {
        'type': 'bounded',
        'vals': [3, 8],
        'guess': 4,
        'pre': int,
    },
    'net.stride': {
        'type': 'bounded',
        'vals': [1, 4],
        'guess': 2,
        'pre': int,
    },
}

# Fill in implicit 'vals' (eg, 'bool' with [True, False])
for _, section in hypers.items():
    for k, v in section.items():
        if type(v) != dict:
            continue  # hard-coded vals
        if v['type'] == 'bool':
            v['vals'] = [0, 1]


class DotDict(object):
    """
    Utility class that lets you get/set attributes with a dot-seperated string key, like `d = a['b.c.d']` or `a['b.c.d'] = 1`
    """

    def __init__(self, obj):
        self._data = obj
        self.update = self._data.update

    def __getitem__(self, path):
        v = self._data
        for k in path.split('.'):
            if k not in v:
                return None
            v = v[k]
        return v

    def __setitem__(self, path, val):
        v = self._data
        path = path.split('.')
        for i, k in enumerate(path):
            if i == len(path) - 1:
                v[k] = val
                return
            elif k in v:
                v = v[k]
            else:
                v[k] = {}
                v = v[k]

    def to_dict(self):
        return self._data


class HSearchEnv(object):
    """
    This is the "wrapper" environment (the "inner" environment is the one you're testing against, like Cartpole-v0).
    This env's actions are all the hyperparameters (above). The state is nothing (`[1.]`), and a single episode is
    running the inner-env however many episodes (300). The inner's last-few reward avg is outer's one-episode reward.
    That's one run: make inner-env, run 300, avg reward, return that. The next episode will be a new set of
    hyperparameters (actions); run inner-env from scratch using new hypers.
    """

    def __init__(self, agent='ppo_agent', gpu_split=1, net_type='lstm'):
        """
        TODO only tested with ppo_agent. There's some code for dqn_agent, but I haven't tested. Nothing else
        is even attempted implemtned
        """
        hypers_ = hypers[agent].copy()
        hypers_.update(hypers['custom'])
        hypers_['net.type'] = net_type  # set as hard-coded val
        hypers_.update(hypers[net_type])

        hardcoded = {}
        for k, v in hypers_.copy().items():
            if type(v) != dict: hardcoded[k] = v

        self.hypers = hypers_
        self.agent = agent
        self.hardcoded = hardcoded
        self.gpu_split = gpu_split
        self.net_type = net_type
        self.conn = data.engine.connect()

    def close(self):
        self.conn.close()

    def get_hypers(self, actions):
        """
        Bit of confusing logic here where I construct a `flat` dict of hypers from the actions - looks like how hypers
        are specified above ('dot.key.str': val). Then from that we hydrate it as a proper config dict (hydrated).
        Keeping `flat` around because I save the run to the database so can be analyzed w/ a decision tree
        (for feature_importances and the like) and that's a good format, rather than a nested dict.
        :param actions: the hyperparamters
        """
        self.flat = flat = {}
        # Preprocess hypers
        for k, v in actions.items():
            try:
                v = v.item()  # sometimes primitive, sometimes numpy
            except Exception:
                pass
            hyper = self.hypers[k]
            if 'pre' in hyper:
                v = hyper['pre'](v)
            flat[k] = v
        flat.update(self.hardcoded)

        # Post-process hypers (allow for dependency handling, etc)
        for k, v in flat.items():
            hyper = self.hypers[k]
            if type(hyper) == dict and 'post' in hyper:
                flat[k] = hyper['post'](v, flat)

        # change all a.b=c to {a:{b:c}} (note DotDict class above, I hate and would rather use an off-the-shelf)
        main, custom = DotDict({}), DotDict({})
        for k, v in flat.items():
            obj = main if k in hypers[self.agent] else custom
            try:
                obj.update(self.hypers[k]['hydrate'](v, self.flat))
            except:
                obj[k] = v
        main, custom = main.to_dict(), custom.to_dict()

        network = build_net_spec(custom)
        if flat['baseline_mode']:
            main['baseline']['network_spec'] = build_net_spec(custom, baseline=True)

        # GPU split
        session_config = None
        if self.gpu_split != 1:
            fraction = .90 / self.gpu_split if self.gpu_split > 1 else self.gpu_split
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=fraction))
        main['session_config'] = session_config

        print('--- Flat ---')
        pprint(flat)
        print('--- Hydrated ---')
        pprint(main)

        return flat, main, network

    def execute(self, actions):
        flat, hydrated, network = self.get_hypers(actions)

        env = OpenAIGym('CartPole-v0', visualize=True)
        env.viewer = None
        agent = agents_dict[self.agent](
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=network,
            **hydrated
        )

        # n_train, n_test = 2, 1
        n_train, n_test = 250, 30
        runner = Runner(agent=agent, environment=env)
        runner.run(episodes=n_train)  # train
        runner.run(episodes=n_test, deterministic=True)  # test
        # You may need to remove runner.py's close() calls so you have access to runner.episode_rewards, see
        # https://github.com/lefnire/tensorforce/commit/976405729abd7510d375d6aa49659f91e2d30a07

        # I personally save away the results so I can play with them manually w/ scikitlearn & SQL
        rewards = runner.episode_rewards
        reward = np.mean(rewards[-n_test:])
        print(flat, f"\nReward={reward}\n\n")

        sql = """
          INSERT INTO runs (hypers, reward_avg, rewards, agent, flag)
          VALUES (:hypers, :reward_avg, :rewards, :agent, :flag)
        """
        try:
            self.conn.execute(
                text(sql),
                hypers=json.dumps(flat),
                reward_avg=reward,
                rewards=rewards,
                agent='ppo_agent',
                flag=self.net_type
            )
        except Exception as e:
            pdb.set_trace()

        runner.close()
        return reward

    def get_winner(self, from_db=True):
        if from_db:
            sql = "SELECT id, hypers FROM runs WHERE agent=:agent ORDER BY reward_avg DESC LIMIT 1"
            winner = self.conn.execute(text(sql), agent=self.agent).fetchone()
            print(f'Using winner {winner.id}')
            winner = winner.hypers
        else:
            winner = {}
            for k, v in self.hypers.items():
                if k not in self.hardcoded:
                    winner[k] = v['guess']
            winner.update(self.hardcoded)
        self.hardcoded = winner
        return self.get_hypers({})


def print_feature_importances(X, Y, feat_names):
    if len(X) < 5: return
    model = GradientBoostingRegressor()
    model_hypers = {
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [None, 10, 20],
        'n_estimators': [100, 200, 300],
    }
    model = GridSearchCV(model, param_grid=model_hypers, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    model.fit(X, np.squeeze(Y))
    feature_imp = sorted(zip(model.best_estimator_.feature_importances_, feat_names), key=lambda x: x[0],
                         reverse=True)
    print('\n\n--- Feature Importances ---\n')
    print('\n'.join([f'{x[1]}: {round(x[0],4)}' for x in feature_imp]))


def main_gp():
    import gp, GPyOpt
    from sklearn.feature_extraction import DictVectorizer

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', type=str, default='ppo_agent', help="Agent to use (ppo_agent|dqn_agent|etc)")
    parser.add_argument('-g', '--gpu_split', type=float, default=1,
                        help="Num ways we'll split the GPU (how many tabs you running?)")
    parser.add_argument('-n', '--net_type', type=str, default='lstm', help="(lstm|conv2d) Which network arch to use")
    parser.add_argument('--guess', action="store_true", default=False,
                        help="Run the hard-coded 'guess' values first before exploring")
    parser.add_argument('--gpyopt', action="store_true", default=False,
                        help="Use GPyOpt library, or use basic sklearn GP implementation? GpyOpt shows more promise, but has bugs.")
    args = parser.parse_args()

    # Encode features
    hsearch = HSearchEnv(gpu_split=args.gpu_split, net_type=args.net_type)
    hypers_, hardcoded = hsearch.hypers, hsearch.hardcoded
    hypers_ = {k: v for k, v in hypers_.items() if k not in hardcoded}
    hsearch.close()

    # Build a matrix of features,  length = max feature size
    max_num_vals = 0
    for v in hypers_.values():
        l = len(v['vals'])
        if l > max_num_vals: max_num_vals = l
    empty_obj = {k: None for k in hypers_}
    mat = pd.DataFrame([empty_obj.copy() for _ in range(max_num_vals)])
    for k, hyper in hypers_.items():
        for i, v in enumerate(hyper['vals']):
            mat.loc[i, k] = v
    mat.ffill(inplace=True)

    # Above is Pandas-friendly stuff, now convert to sklearn-friendly & pipe through OneHotEncoder
    vectorizer = DictVectorizer()
    vectorizer.fit(mat.T.to_dict().values())
    feat_names = vectorizer.get_feature_names()

    # Map TensorForce actions to GPyOpt-compatible `domain`
    # instantiate just to get actions (get them from hypers above?)
    bounds = []
    for k in feat_names:
        hyper = hypers_.get(k, False)
        if hyper:
            bounded, min_, max_ = hyper['type'] == 'bounded', min(hyper['vals']), max(hyper['vals'])
        if args.gpyopt:
            b = {'name': k, 'type': 'discrete', 'domain': (0, 1)}
            if bounded: b.update(type='continuous', domain=(min_, max_))
        else:
            b = [min_, max_] if bounded else [0, 1]
        bounds.append(b)

    def hypers2vec(obj):
        h = dict()
        for k, v in obj.items():
            if k in hardcoded: continue
            if type(v) == bool:
                h[k] = float(v)
            else:
                h[k] = v or 0.
        return vectorizer.transform(h).toarray()[0]

    def vec2hypers(vec):
        # Reverse the encoding
        # https://stackoverflow.com/questions/22548731/how-to-reverse-sklearn-onehotencoder-transform-to-recover-original-data
        # https://github.com/scikit-learn/scikit-learn/issues/4414
        if not args.gpyopt: vec = [vec]  # gp.py passes as flat, GPyOpt as wrapped
        reversed = vectorizer.inverse_transform(vec)[0]
        obj = {}
        for k, v in reversed.items():
            if '=' not in k:
                obj[k] = v
                continue
            if k in obj: continue  # we already handled this x=y logic (below)
            # Find the winner (max) option for this key
            score, attr, val = v, k.split('=')[0], k.split('=')[1]
            for k2, score2 in reversed.items():
                if k2.startswith(attr + '=') and score2 > score:
                    score, val = score2, k2.split('=')[1]
            obj[attr] = val

        # Bools come in as floats. Also, if the result is False they don't come in at all! So we start iterate
        # hypers now instead of nesting this logic in reversed-iteration above
        for k, v in hypers_.items():
            if v['type'] == 'bool':
                obj[k] = bool(round(obj.get(k, 0.)))
        return obj

    # Specify the "loss" function (which we'll maximize) as a single rl_hsearch instantiate-and-run
    def loss_fn(params):
        hsearch = HSearchEnv(gpu_split=args.gpu_split, net_type=args.net_type)
        reward = hsearch.execute(vec2hypers(params))
        hsearch.close()
        return [reward]

    while True:
        conn = data.engine.connect()
        sql = "SELECT hypers, reward_avg FROM runs WHERE flag=:f"
        runs = conn.execute(text(sql), f=args.net_type).fetchall()
        conn.close()
        X, Y = [], []
        for run in runs:
            X.append(hypers2vec(run.hypers))
            Y.append([run.reward_avg])
        print_feature_importances(X, Y, feat_names)

        if args.guess:
            guesses = {k: v['guess'] for k, v in hypers_.items()}
            X.append(hypers2vec(guesses))
            Y.append([None])
            args.guess = False

        if args.gpyopt:
            pretrain = {'X': np.array(X), 'Y': np.array(Y)} if X else {}
            opt = GPyOpt.methods.BayesianOptimization(
                f=loss_fn,
                domain=bounds,
                maximize=True,
                **pretrain
            )
            # using max_iter=1 because of database setup. Normally you'd go until convergence, but since we're using
            # a database for the runs we can parallelize runs across machines (connected to the same database). Then
            # between each run we can grab the result from the other machines and merge with our own; so only run
            # once, reset the model-fitting w/ the full database (which may have grown), and repeat
            opt.run_optimization(max_iter=1)
        else:
            gp.bayesian_optimisation2(
                n_iters=1,
                loss_fn=loss_fn,
                bounds=np.array(bounds),
                x_list=X,
                y_list=Y
            )


if __name__ == '__main__':
    main_gp()
