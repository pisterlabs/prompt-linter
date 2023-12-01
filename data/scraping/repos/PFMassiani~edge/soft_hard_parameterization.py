import numpy as np
import logging
logger = logging.getLogger(__name__)
import gym

from edge.envs import Slip, Hovership
from edge.gym_wrappers import GymEnvironmentWrapper
from edge.reward import AffineReward, ConstantReward
from edge.agent import Agent
from edge.model.value_models import GPQLearning
from edge.model.safety_models import MaternSafety
from edge.model.policy_models import ConstrainedEpsilonGreedy, \
    SafetyMaximization, SafetyActiveSampling


class LowGoalSlip(Slip):
    # * This goal incentivizes the agent to run fast
    def __init__(self, dynamics_parameters=None, reward_done_threshold=None):
        super(LowGoalSlip, self).__init__(
            dynamics_parameters=dynamics_parameters,
            reward_done_threshold=reward_done_threshold,
            random_start=True
        )

        reward = AffineReward(self.stateaction_space, [(1, 0), (0, 0)])
        self.reward = reward


class PenalizedSlip(LowGoalSlip):
    def __init__(self, penalty_level=100, dynamics_parameters=None,
                 reward_done_threshold=None):
        super(PenalizedSlip, self).__init__(dynamics_parameters,
                                            reward_done_threshold)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class LowGoalHovership(Hovership):
    def __init__(self, dynamics_parameters=None, reward_done_threshold=None):
        super(LowGoalHovership, self).__init__(
            random_start=True,
            dynamics_parameters=dynamics_parameters,
            reward_done_threshold=reward_done_threshold,
        )

        reward = AffineReward(self.stateaction_space, [(10, 0), (0, 0)])
        self.reward = reward


class PenalizedHovership(LowGoalHovership):
    def __init__(self, penalty_level=100, dynamics_parameters=None,
                 reward_done_threshold=None):
        super(PenalizedHovership, self).__init__(dynamics_parameters,
                                                 reward_done_threshold)

        def penalty_condition(state, action, new_state, reward):
            return self.is_failure_state(new_state)

        penalty = ConstantReward(self.reward.stateaction_space, -penalty_level,
                                 reward_condition=penalty_condition)

        self.reward += penalty


class CartPole(GymEnvironmentWrapper):
    def __init__(self, discretization_shape, control_frequency):
        gym_env = gym.make('CartPole-v1')
        super(CartPole, self).__init__(gym_env, discretization_shape,
                                       control_frequency=control_frequency)

    @property
    def in_failure_state(self):
        # This condition is taken from OpenAI Gym documentation
        failed = np.abs(self.s[0]) > 2.4  # Linear position
        failed |= np.abs(self.s[2]) > 12 * np.pi / 180  # Angular position
        return failed

    def linearization(self, discrete_time=True):
        # The full equations are taken from
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # which is the source given by OpenAI Gym for the dynamics
        g = self.gym_env.gravity
        m = self.gym_env.total_mass
        eta = self.gym_env.masspole / m
        l = self.gym_env.length * 2  # Gym stores half the pole's length
        a12 = g * eta / (eta - 4 / 3)
        a32 = (g / l) / (4 / 3 - eta)
        A = np.array([[0, 1, 0, 0],
                      [0, 0, a12, 0],
                      [0, 0, 0, 1],
                      [0, 0, a32, 0]])
        B = (1 / m) * np.array([0,
                                (4 / 3) / (4 / 3 - eta),
                                0,
                                -1 / l / (4 / 3 - eta)]).reshape((-1, 1))
        # The action we give to Gym is simply +- 1: we need to multiply B
        # by the true magnitude of the force so we can feed it actions directly
        B = self.gym_env.force_mag * B

        if discrete_time:
            # A and B are the dynamics matrices of the _continuous_ dynamics.
            # We need to apply the integration scheme to get the true matrices
            integrator = self.gym_env.kinematics_integrator
            if integrator != 'euler':
                logger.warning(
                    f'Kinematics integrator is {integrator}, but only '
                    f'euler is supported.')

            tau = self.control_frequency if self.control_frequency is not None \
                else self.gym_env.tau
            A = np.eye(A.shape[0], dtype=float) + tau * A
            B = tau * B

        return A, B


class LunarLander(GymEnvironmentWrapper):
    def __init__(self, discretization_shape):
        gym_env = gym.make('LunarLanderContinuous-v2')
        super(LunarLander, self).__init__(gym_env, discretization_shape)


class SoftHardLearner(Agent):
    """
        Defines an Agent modelling the Q-Values with a MaternGP updated with the Q-Learning update. The Agent also has a
        model for the underlying safety measure with a SafetyModel, and the Q-Learning update is constrained to stay in the
        current estimate of the safe set. If the update chosen by Q-Learning is within a conservative estimate of the
        viable set, then the sample is used to update the safety model.
        """

    def __init__(self, env,
                 greed, step_size, discount_rate, q_x_seed, q_y_seed,
                 gamma_optimistic, gamma_hard, lambda_hard, gamma_soft, s_x_seed, s_y_seed,
                 q_gp_params=None, s_gp_params=None, keep_seed_in_data=True):
        """
        Initializer
        :param env: the environment
        :param greed: the epsilon parameter of the ConstrainedEpsilonGreedy policy
        :param q_step_size: the step size in the Q-Learning update
        :param discount_rate: the discount rate
        :param q_x_seed: the seed input of the GP for the Q-Values model
        :param q_y_seed: the seed output of the GP for the Q-Values model
        :param gamma_optimistic: the gamma parameter for Q_optimistic
        :param gamma_hard: the gamma parameter for Q_hard, the set where Q-Learning is constrained (~ Q_cautious)
        :param lambda_hard: the lambda parameter for Q_hard AND Q_soft
        :param gamma_soft: the gamma parameter for Q_soft, the set outside of which the safety measure is updated
        :param s_x_seed: the seed input of the GP for the safety model
        :param s_y_seed: the seed output of the GP for the safety model
        :param q_gp_params: the parameters defining the GP for the Q-Values model. See edge.models.inference.MaternGP
            for more information
        :param q_gp_params: the parameters defining the GP for the safety model. See edge.models.inference.MaternGP
            for more information
        :param keep_seed_in_data: whether to keep the seed data in the GPs datasets. Should be True, otherwise GPyTorch
            fails.
        """
        Q_model = GPQLearning(env.stateaction_space, step_size, discount_rate,
                              x_seed=q_x_seed, y_seed=q_y_seed,
                              gp_params=q_gp_params)
        safety_model = MaternSafety(env.stateaction_space, gamma_optimistic,
                                    x_seed=s_x_seed, y_seed=s_y_seed,
                                    gp_params=s_gp_params)
        super(SoftHardLearner, self).__init__(env, Q_model, safety_model)

        self.Q_model = Q_model
        self.safety_model = safety_model
        self.lambda_hard = lambda_hard
        self.gamma_hard = gamma_hard
        self.gamma_soft = gamma_soft
        self._gamma_optimistic = gamma_optimistic

        self.constrained_value_policy = ConstrainedEpsilonGreedy(
            self.env.stateaction_space, greed)
        self.safety_maximization_policy = SafetyMaximization(
            self.env.stateaction_space)
        self.active_sampling_policy = SafetyActiveSampling(
            self.env.stateaction_space)

        self.keep_seed_in_data = keep_seed_in_data
        if not keep_seed_in_data:
            self.Q_model.empty_data()

        self.violated_soft_constraint = None
        self.updated_safety = None

    @property
    def greed(self):
        """
        Returns the epsilon parameter of the ConstrainedEpsilonGreedy policy
        :return: epsilon parameter
        """
        return self.constrained_value_policy.greed

    @greed.setter
    def greed(self, new_greed):
        """
        Sets the epsilon parameter of the ConstrainedEpsilonGreedy policy
        """
        self.constrained_value_policy.greed = new_greed

    @property
    def step_size(self):
        """
        Returns the step_size parameter of the Q-Learning model
        :return: step_size parameter
        """
        return self.Q_model.step_size

    @step_size.setter
    def step_size(self, new_step_size):
        """
        Sets the epsilon parameter of the ConstrainedEpsilonGreedy policy
        """
        self.Q_model.step_size = new_step_size

    @property
    def gamma_optimistic(self):
        return self._gamma_optimistic

    @gamma_optimistic.setter
    def gamma_optimistic(self, new_gamma_optimistic):
        self._gamma_optimistic = new_gamma_optimistic
        self.safety_model.gamma_measure = new_gamma_optimistic

    def get_next_action(self):
        is_cautious_list, proba_slice_list = self.safety_model.level_set(
            self.state,
            lambda_threshold=[self.lambda_hard, self.lambda_hard],
            gamma_threshold=[self.gamma_hard, self.gamma_soft],
            return_proba=True,
            return_covar=False
        )
        is_cautious_hard, is_cautious_soft = is_cautious_list
        is_cautious_hard = is_cautious_hard.squeeze()
        is_cautious_soft = is_cautious_soft.squeeze()
        proba_slice_hard = proba_slice_list[0].squeeze()

        q_values = self.Q_model[self.state, :]
        action = self.constrained_value_policy.get_action(
            q_values, is_cautious_hard
        )

        if action is None:
            logger.info('No viable action, taking the safest...')
            action = self.safety_maximization_policy.get_action(proba_slice_hard)
            self.violated_soft_constraint = True
        else:
            action_idx = self.env.action_space.get_index_of(action)
            self.violated_soft_constraint = not is_cautious_soft[action_idx]

        return action

    def get_random_safe_state(self):
        """
        Returns a random state that is classified as safe by the safety model
        :return: a safe state
        """
        is_viable = self.safety_model.measure(
            slice(None, None, None),
            lambda_threshold=self.lambda_hard,
            gamma_threshold=self.gamma_hard
        ) > 0
        viable_indexes = np.atleast_1d(np.argwhere(is_viable).squeeze())
        try:
            state_index = viable_indexes[np.random.choice(len(viable_indexes))]
        except Exception as e:
            logger.error('ERROR:', str(e))
            return None
        state = self.env.stateaction_space.state_space[state_index]
        return state

    def update_models(self, state, action, next_state, reward, failed):
        self.Q_model.update(state, action, next_state, reward, failed)
        self.safety_model.update(state, action, next_state, reward, failed)

    def fit_models(self, q_train_x=None, q_train_y=None, q_epochs=None, q_optimizer_kwargs=None,
                   s_train_x=None, s_train_y=None, s_epochs=None, s_optimizer_kwargs=None):
        if q_train_x is not None:
            if q_optimizer_kwargs is None:
                q_optimizer_kwargs = {}
            self.Q_model.fit(q_train_x, q_train_y, q_epochs, **q_optimizer_kwargs)
            if not self.keep_seed_in_data:
                self.Q_model.empty_data()

        if s_train_x is not None:
            if s_optimizer_kwargs is None:
                s_optimizer_kwargs = {}
            self.safety_model.fit(s_train_x, s_train_y, s_epochs, **s_optimizer_kwargs)
            if not self.keep_seed_in_data:
                self.safety_model.empty_data()

    def step(self):
        """
        Chooses an action according to the policy, takes a step in the Environment, and updates the models. The action
        taken is available in self.last_action.
        :return: new_state, reward, failed
        """
        old_state = self.state
        action = self.get_next_action()
        new_state, reward, failed = self.env.step(action)
        if self.violated_soft_constraint or failed:
            self.Q_model.update(old_state, action, new_state, reward, failed)
            self.safety_model.update(old_state, action, new_state, reward, failed)
            self.updated_safety = True
        else:
            self.Q_model.update(old_state, action, new_state, reward, failed)
            self.updated_safety = False
        self.state = new_state
        self.last_action = action
        return new_state, reward, failed, self.env.done