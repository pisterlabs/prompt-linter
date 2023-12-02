import numpy as np
import gym
from epidemioptim.environments.gym_envs.base_env import BaseEnv


class EpidemicDiscrete(BaseEnv):
    def __init__(self,
                 cost_function,
                 model,
                 simulation_horizon,
                 ratio_death_to_R=0.005,  # death ratio among people who were infected
                 time_resolution=7,
                 seed=np.random.randint(1e6)
                 ):
        """
        EpidemicDiscrete environment is based on the Epidemiological SEIRAH model from Prague et al., 2020 and on a bi-objective
        cost function (death toll and gdp recess).

        Parameters
        ----------
        cost_function: BaseCostFunction
            A cost function.
        model: BaseModel
            An epidemiological model.
        simulation_horizon: int
            Simulation horizon in days.
        ratio_death_to_R: float
            Ratio of deaths among recovered individuals.
        time_resolution: int
            In days.
        """

        # Initialize model
        self.model = model
        self.stochastic = self.model.stochastic
        self.simulation_horizon = simulation_horizon
        self.reset_same = False  # whether the next reset resets the same epidemiological model

        # Initialize cost function
        self.cost_function = cost_function
        self.nb_costs = cost_function.nb_costs
        self.cumulative_costs = [0 for _ in range(self.nb_costs)]

        # Initialize states
        self.state_labels = self.model.internal_states_labels + ['previous_lockdown_state', 'current_lockdown_state'] + \
            ['cumulative_cost_{}'.format(id_cost) for id_cost in range(self.cost_function.nb_costs)] + ['level_b']
        self.label_to_id = dict(zip(self.state_labels, np.arange(len(self.state_labels))))
        self.normalization_factors = [self.model.current_internal_params['N_av']] * len(self.model.internal_states_labels) + \
                                     [1, 1, self.model.current_internal_params['N_av'], 150, 1]

        super().__init__(cost_function=cost_function,
                         model=model,
                         simulation_horizon=simulation_horizon,
                         dim_action=2,
                         discrete=True,
                         seed=seed)

        self.ratio_death_to_R = ratio_death_to_R
        self.time_resolution = time_resolution
        self._max_episode_steps = simulation_horizon // time_resolution
        self.history = None

        # Action modalities
        self.level_b_splits = (7, 14, 21)  # switches between transmission rates, in days (4 stages)
        self.level_b = 0  # index of the stage
        self.b0 = self.model.current_internal_params['b_fit']  # initial transmission rate
        self.betas = [self.b0] + [np.exp(self.model.current_internal_params['beta{}'.format(i + 1)]) for i in range(4)]  # factors of reduction for each stage
        self.bs = None

    def _compute_b(self, times_since_start, times_since_last):
        """
        Computes the transmission rate depending on the number of days since the last lock-down or since beginning of the current lock-down.

        Parameters
        ----------
        times_since_start: nd.array of ints
            Time since the start of the current lock-down, for each day.
        times_since_last: nd.array of ints
            Time since the last lock-down, for each day.

        Returns
        -------
        list
            The values of transmission rates for each day.
        """
        if self.lockdown_state == 0:
            # if new lock-down decrease the transmission rate of one stage
            if self.previous_lockdown_state != self.lockdown_state:
                self.level_b =  max(self.level_b - 1, 0)

            # further decrease the transmission rate every 7 days until first stage.
            assert times_since_start.size == 0
            bs = []
            for t_i in times_since_last:
                if t_i in self.level_b_splits:
                    self.level_b =  max(self.level_b - 1, 0)
                bs.append(np.product(self.betas[:self.level_b + 1]))
        else:
            # if lock-down terminated, increase the transmission rate of one stage.
            if self.previous_lockdown_state != self.lockdown_state:
                self.level_b = min(self.level_b + 1, len(self.betas) - 1)

            # further increase the transmission rate every 7 days until last stage.
            assert times_since_last.size == 0
            bs = []
            for t_i in times_since_start:
                if t_i in self.level_b_splits:
                    self.level_b = min(self.level_b + 1, len(self.betas) - 1)
                bs.append(np.product(self.betas[:self.level_b+1]))
        return bs

    def _update_previous_env_state(self):
        """
        Save previous env state.

        """
        if self.env_state is not None:
            self.previous_env_state = self.env_state.copy()
            self.previous_env_state_labelled = self.env_state_labelled.copy()

    def _update_env_state(self):
        """
        Update the environment state.

        """

        # Update env state
        self.env_state_labelled = dict(zip(self.model.internal_states_labels, self.model_state))
        self.env_state_labelled.update(previous_lockdown_state=self.previous_lockdown_state,
                                       current_lockdown_state=self.lockdown_state,
                                       level_b=self.level_b)
        # track cumulative costs in the state.
        for id_cost in range(self.nb_costs):
            self.env_state_labelled['cumulative_cost_{}'.format(id_cost)] = self.cumulative_costs[id_cost]
        assert sorted(list(self.env_state_labelled.keys())) == sorted(self.state_labels), "labels do not match"
        self.env_state = np.array([self.env_state_labelled[k] for k in self.state_labels])

        # Set previous env state to env state if first step
        if self.previous_env_state is None:
            # happens at first step
            self.previous_env_state = self.env_state.copy()
            self.previous_env_state_labelled = self.env_state_labelled.copy()

    def reset_same_model(self):
        """
        To call if you want to reset to the same model the next time you call reset.
        Will be cancelled after the first reset, it needs to be called again each time.


        """
        self.reset_same = True

    def reset(self):
        """
        Reset the environment and the tracking of data.

        Returns
        -------
        nd.array
            The initial environment state.

        """
        # initialize history of states, internal model states, actions, cost_functions, deaths
        self.history = dict(env_states=[],
                            model_states=[],
                            env_timesteps=[],
                            actions=[],
                            aggregated_costs=[],
                            costs=[],
                            lockdown=[],
                            deaths=[],
                            b=[])
        # initialize time and lockdown days counter
        self.t = 0
        self.count_lockdown = 0
        self.count_deaths = 0
        self.count_since_start_lockdown = 0
        self.count_since_last_lockdown = 0
        self.level_b = 0
        self.b = self.model.current_internal_params['b_fit']

        self.lockdown_state = 0  # 0 not lockdown, 1 lockdown
        self.previous_lockdown_state = self.lockdown_state
        self.cumulative_costs = [0 for _ in range(self.nb_costs)]

        # initialize model internal state and params
        if self.reset_same:
            self.model.reset_same_model()
            self.reset_same = False
        else:
            self.model.reset()
        self.model_state = self.model._get_current_state()

        self._update_previous_env_state()
        self._update_env_state()

        self.history['env_states'].append(self.env_state.copy())
        self.history['model_states'].append(self.model_state.copy().tolist())
        self.history['env_timesteps'].append(self.t)

        return self._normalize_env_state(self.env_state)

    def update_with_action(self, action):
        """
        Implement effect of action on transmission rate.

        Parameters
        ----------
        action: int
            Action is 0 (no lock-down) or 1 (lock-down).

        """

        # Translate actions
        self.previous_lockdown_state = self.lockdown_state
        previous_count_start = self.count_since_start_lockdown
        previous_count_last = self.count_since_last_lockdown

        if action == 0:
            # no lock-down
            self.jump_of = min(self.time_resolution, self.simulation_horizon - self.t)
            self.lockdown_state = 0
            if self.previous_lockdown_state == self.lockdown_state:
                self.count_since_last_lockdown += self.jump_of
            else:
                self.count_since_last_lockdown = self.jump_of
                self.count_since_start_lockdown = 0
        else:
            self.jump_of = min(self.time_resolution, self.simulation_horizon - self.t)
            self.lockdown_state = 1
            if self.lockdown_state == self.previous_lockdown_state:
                self.count_since_start_lockdown += self.jump_of
            else:
                self.count_since_start_lockdown = self.jump_of
                self.count_since_last_lockdown = 0

        # Modify model parameters based on lockdown state
        since_start = np.arange(previous_count_start, self.count_since_start_lockdown)
        since_last = np.arange(previous_count_last, self.count_since_last_lockdown)
        self.bs = self._compute_b(times_since_start=since_start, times_since_last=since_last)
        self.model.current_internal_params['b_fit'] = self.b

    def step(self, action):
        """
        Traditional step function from OpenAI Gym envs. Uses the action to update the environment.

        Parameters
        ----------
        action: int
            Action is 0 (no lock-down) or 1 (lock-down).


        Returns
        -------
        state: nd.array
            New environment state.
        cost_aggregated: float
            Aggregated measure of the cost.
        done: bool
            Whether the episode is terminated.
        info: dict
            Further infos. In our case, the costs, icu capacity of the region and whether constraints are violated.

        """
        action = int(action)
        assert 0 <= action < self.dim_action

        self.update_with_action(action)
        if self.lockdown_state == 1:
            self.count_lockdown += self.jump_of

        # Run model for jump_of steps
        model_state = [self.model_state]
        model_states = []
        for b in self.bs:
            self.model.current_internal_params['b_fit'] = b
            model_state = self.model.run_n_steps(model_state[-1], 1)
            model_states += model_state.tolist()
        self.model_state = model_state[-1]  # last internal state is the new current one
        self.t += self.jump_of

        # Update state
        self._update_previous_env_state()
        self._update_env_state()

        # Store history
        costs = [c.compute_cost(previous_state=np.atleast_2d(self.previous_env_state),
                                state=np.atleast_2d(self.env_state),
                                label_to_id=self.label_to_id,
                                action=action,
                                others=dict(jump_of=self.time_resolution))[0] for c in self.cost_function.costs]
        for i in range(len(costs)):
            self.cumulative_costs[i] += costs[i]
        n_deaths = self.cost_function.compute_deaths(previous_state=np.atleast_2d(self.previous_env_state),
                                                     state=np.atleast_2d(self.env_state),
                                                     label_to_id=self.label_to_id,
                                                     action=action)[0]

        self._update_env_state()

        self.history['actions'] += [action] * self.jump_of
        self.history['env_states'] += [self.env_state.copy()] * self.jump_of
        self.history['env_timesteps'] += list(range(self.t - self.jump_of, self.t))
        self.history['model_states'] += model_states
        self.history['lockdown'] += [self.lockdown_state] * self.jump_of
        self.history['deaths'] += [n_deaths / self.jump_of] * self.jump_of
        self.history['b'] += self.bs

        # Compute cost_function
        cost_aggregated, costs, over_constraints = self.cost_function.compute_cost(previous_state=self.previous_env_state,
                                                                                   state=self.env_state,
                                                                                   label_to_id=self.label_to_id,
                                                                                   action=action,
                                                                                   others=dict(jump_of=self.jump_of))
        costs = costs.flatten()

        self.history['aggregated_costs'] += [cost_aggregated / self.jump_of] * self.jump_of
        self.history['costs'] += [costs / self.jump_of for _ in range(self.jump_of)]
        self.costs = costs.copy()

        if self.t >= self.simulation_horizon:
            done = 1
        else:
            done = 0

        return self._normalize_env_state(self.env_state), cost_aggregated, done, dict(costs=costs,
                                                                                      constraints=over_constraints.flatten(),
                                                                                      n_icu=self.env_state[self.label_to_id["H"]] * 0.25)

    # Utils
    def _normalize_env_state(self, env_state):
        return (env_state / np.array(self.normalization_factors)).copy()

    def _set_rew_params(self, goal):
        self.cost_function.set_goal_params(goal.copy())

    def sample_cost_function_params(self):
        return self.cost_function.sample_goal_params()

    # Format data for plotting
    def get_data(self):

        data = dict(history=self.history.copy(),
                    time_jump=1,
                    model_states_labels=self.model.internal_states_labels,
                    icu_capacity=self.model.current_internal_params['icu'])
        t = self.history['env_timesteps']
        cumulative_death = [np.sum(self.history['deaths'][:i]) for i in range(len(t) - 1)]
        cumulative_eco_cost = [np.array(self.history['costs'])[:i, 1].sum() for i in range(len(t) - 1)]
        betas = [0, 0.25, 0.5, 0.75, 1]
        costs = np.array(self.history['costs'])
        aggregated = [self.cost_function.compute_aggregated_cost(costs, beta) for beta in betas]
        to_plot = [np.array(self.history['deaths']),
                   np.array(cumulative_death),
                   aggregated,
                   costs[:, 1],
                   np.array(cumulative_eco_cost),
                   np.array(self.history['b'])
                   ]
        labels = ['New Deaths', 'Total Deaths', r'Aggregated Cost', 'New GDP Loss (B)', 'Total GDP Loss (B)', 'Transmission rate']
        legends = [None, None, [r'$\beta = $' + str(beta) for beta in betas], None, None, None]
        stats_run = dict(to_plot=to_plot,
                         labels=labels,
                         legends=legends)
        data['stats_run'] = stats_run
        data['title'] = 'Eco cost: {:.2f} B, Death Cost: {}, Aggregated Cost: {:.2f}'.format(cumulative_eco_cost[-1],
                                                                                             int(cumulative_death[-1]),
                                                                                             np.sum(self.history['aggregated_costs']))
        return data


if __name__ == '__main__':
    from epidemioptim.utils import plot_stats
    from epidemioptim.environments.cost_functions import get_cost_function
    from epidemioptim.environments.models import get_model

    simulation_horizon = 364
    stochastic = False
    region = 'IDF'

    model = get_model(model_id='prague_seirah', params=dict(region=region,
                                                      stochastic=stochastic))

    N_region = model.pop_sizes[region]
    N_country = np.sum(list(model.pop_sizes.values()))
    ratio_death_to_R = 0.005

    cost_func = get_cost_function(cost_function_id='multi_cost_death_gdp_controllable', params=dict(N_region=N_region,
                                                                                                    N_country=N_country,
                                                                                                    ratio_death_to_R=ratio_death_to_R)
                                  )

    env = gym.make('EpidemicDiscrete-v0',
                   cost_function=cost_func,
                   model=model,
                   simulation_horizon=simulation_horizon)
    env.reset()

    actions = np.random.choice([0, 1], size=53)
    actions = np.zeros([53])
    actions[3:3+8] = 1
    t = 0
    r = 0
    done = False
    while not done:
        out = env.step(actions[t])
        t += 1
        r += out[1]
        done = out[2]
    stats = env.unwrapped.get_data()

    # plot model states
    plot_stats(t=stats['history']['env_timesteps'],
               states=np.array(stats['history']['model_states']).transpose(),
               labels=stats['model_states_labels'],
               lockdown=np.array(stats['history']['lockdown']),
               icu_capacity=stats['icu_capacity'],
               time_jump=stats['time_jump'])
    plot_stats(t=stats['history']['env_timesteps'][1:],
               states=stats['stats_run']['to_plot'],
               labels=stats['stats_run']['labels'],
               legends=stats['stats_run']['legends'],
               title=stats['title'],
               lockdown=np.array(stats['history']['lockdown']),
               time_jump=stats['time_jump'],
               show=True
               )
