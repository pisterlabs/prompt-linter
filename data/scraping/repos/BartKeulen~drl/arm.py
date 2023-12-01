import numpy as np
from drl.utilities import rk_step
from abc import abstractmethod, ABCMeta
from gym import spaces
from drl.env.environment import Environment


class Arm(Environment, metaclass=ABCMeta):
    """
    Arm parent class for creating robotic arm simulation.
    On top of this class robotics arms with various number of links can easily be created.
    """

    def __init__(self, name, dof, g=0., dt=0.05, wp=10., wv=1., wu=0.001, action_high=None, velocity_high=None):
        """
        Constructs a new 'Arm' object.

        :param dof: degree of freedoms of the robotics arm
        :param g: gravity
        :param dt: time-step
        :param action_high: action limit (assumed lower limit is the same as high limit)
        :param velocity_high: velocity limit (assumed lower limit is the same as high limit)
        """
        super(Arm, self).__init__(name)
        self.params = (g,)
        self.dt = dt
        self.dof = dof
        self.state_dim = 2 * self.dof
        self.action_dim = self.dof

        self.wp = wp
        self.wv = wv
        self.wu = wu

        self.new_goal = False

        self.viewer = None

        if action_high is None:
            self.action_high = np.ones(self.action_dim)
        else:
            self.action_high = action_high

        if velocity_high is None:
            self.velocity_high = np.ones(self.dof)
        else:
            self.velocity_high = velocity_high

        # obs_high = np.concatenate([
        #     np.ones(self.dof),
        #     self.velocity_high,
        #     np.array([100.])
        # ])

        # self.observation_space = spaces.Box(low=-obs_high, high=obs_high)
        self.action_space = spaces.Box(low=-self.action_high, high=self.action_high)

    def reset(self, q=None, goal=None, full_state=False):
        """
        Reset the environment.

        :param q: if given reset to q else random
        :param goal: if given reset to goal else random
        :return: new observation / new state
        """
        if q is None:
            self.q = np.pad(np.random.random_sample(self.dof)*2*np.pi - np.pi, (0, self.dof), 'constant')
        else:
            self.q = q
        self.set_goal(goal=goal)
        if full_state:
            return self.q
        else:
            return self._get_obs()

    def set_goal(self, goal=None):
        """
        Set the goal of the environment.

        :param goal: if given set to goal else random
        """
        if goal is None:
            self.goal = np.pad(np.random.random_sample(self.dof) * 2 * np.pi - np.pi, (0, self.dof), 'constant')
        else:
            self.goal = goal
        self.new_goal = True

    def get_goal(self, cartesian=False):
        """
        Return the goal.

        :param cartesian: if the goal is returned in angles or cartesian coordinates
        :return: goal
        """
        if cartesian:
            x = self.to_cartesian(self.goal)
            return np.array(x[-2:])
        else:
            return self.goal

    def step(self, u, full_state=False):
        """
        Take a step by executing control input u.

        :param u: control input
        :return: new observation / new state, reward, terminal, {}
        """
        u = np.clip(u, -self.action_high, self.action_high)

        q, _ = self.dynamics_func(self.q, u)
        self.q = self.clip_state(q)

        r = self.reward_func(self.q, u)
        t = self.terminal_func(self.q, u)

        if full_state:
            return self.q, r, t, {}
        else:
            return self._get_obs(), r, t, {}

    def dynamics_func(self, q, u):
        """
        Execute dynamics from state q by executing control input u

        :param q: state
        :param u: control input
        :return: new state, state derivative
        """
        qnew, qdot = rk_step(self._eom, q, u, self.dt)
        return self.clip_state(qnew), qdot

    def cost_func(self, q, u):
        """
        Calculates the cost for state q and control input u.
        If is is set to nan the final cost will be calculated.

        The cost consists of three parts:
            * lu - cost on the control input
            * lp - cost on position (Euclidean distance between goal and end effector)
            * lv - cost on velocity

        lp, lv are only added for final position

        :param q: state
        :param u: control input, set to nan to calculate final cost
        :return: cost
        """
        final = np.isnan(u)
        u[final] = 0

        lu = self.wu*np.sum(u*u)

        if final.any():
            d = self._distance(q)
            lp = self.wp * d
            lv = self.wv * np.sum(q[self.dof:] * q[self.dof:])
        else:
            lp = 0
            lv = 0

        c = lu + lp + lv
        return c, None, None, None, None, None

    def reward_func(self, q, u):
        """
        Reward function is the negative cost.

        :param q: state
        :param u: control input
        :return: reward
        """
        c, _, _, _, _, _ = self.cost_func(q, np.full([1, self.action_dim], np.nan))
        return -c

    def terminal_func(self, q, u):
        """
        Terminal function determines if the agent is in a terminal state or not.

        :param q: state
        :param u: control input
        :return: True if terminal otherwise False
        """
        d = self._distance(q)
        if np.abs(d) < 0.05 and np.sum(q[self.dof:]*q[self.dof:]) < 0.01:
            return True
        else:
            return False

    def _distance(self, q):
        """
        Calculates the euclidean distance between the goal and end effector.

        :param q: state
        :return: distance
        """
        goal = self.to_cartesian(self.goal)
        x = self.to_cartesian(q)
        return np.sqrt((x[-2] - goal[-2]) ** 2 + (x[-1] - goal[-1]) ** 2)

    @abstractmethod
    def _eom(self, q, u):
        """
        Placeholder for equations of motion function. Must be defined in child classes.

        :param q: state
        :param u: control input
        :return: Exception: 'Arm cannot be run as standalone class, define child class with the equations of motion.'
        """

    def to_cartesian(self, q):
        """
        Converts state to Cartesian coordinates

        :param q: state
        :return: cartesian coordinates (x1, y1, x2, y2, ..., xn, yn)
        """
        theta = q[0]
        x = (np.sin(theta)*self.params[2], np.cos(theta)*self.params[2])
        for i in range(1, self.dof):
            theta += q[i]
            x += (x[i*2-2] + np.sin(theta)*self.params[i*2+2], x[i*2-1] + np.cos(theta)*self.params[i*2+2])
        return x

    def clip_state(self, q):
        """
        Clip state:
            * Convert angles to lay between -2*pi and 2*pi
            * Clip velocities between -velocity_high and velocity_high

        :param q: state
        :return: clipped state
        """
        for i in range(self.dof):
            if q[i] > 2 * np.pi:
                q[i] -= 2 * np.pi
            elif q[i] < -2 * np.pi:
                q[i] += 2 * np.pi

        q[self.dof:] = np.clip(q[self.dof:], -self.velocity_high, self.velocity_high)

        return q

    def _get_obs(self):
        """
        Converts the current state to observation.
        Observation consists of cos and sin of joint angles, join angular velocities and euclidean distance to goal.

        :return: new observation
        """
        return np.concatenate([
            np.array(self.to_cartesian(self.q)),
            self.q[-2:],
            np.array(self.get_goal(True))
        ])
        # return np.concatenate([
        #     self.q,
        #     np.array([self._distance(self.q)])
        # ])

    def render(self, mode='human', close=False):
        """
        Renders the environment.
        Copied the rendering structure from OpenAI gym (https://gym.openai.com/)

        :param mode:
        :param close: if True the viewer is closed
        :return:
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            l = []
            for i in range(self.dof):
                l.append(self.params[2+i*2])
            size = np.sum(l) + 0.1
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-size, size, -size, size)

            goal = rendering.make_circle(0.1)
            goal.set_color(0., .8, .2)
            self.goal_transform = rendering.Transform()
            goal.add_attr(self.goal_transform)
            self.viewer.add_geom(goal)

            self.poles = []
            self.joints = []
            for i in range(self.dof+1):
                if i < self.dof:
                    pole = rendering.make_capsule(l[i], .1)
                    pole.set_color(.5, .5, .5)
                    pole_transform = rendering.Transform()
                    pole.add_attr(pole_transform)
                    self.poles.append(pole_transform)
                    self.viewer.add_geom(pole)

                joint = rendering.make_circle(0.04)
                joint.set_color(0., 0., 0.)
                joint_transform = rendering.Transform()
                joint.add_attr(joint_transform)
                self.joints.append(joint_transform)
                self.viewer.add_geom(joint)

        if self.new_goal:
            x = self.to_cartesian(self.goal)
            goalx, goaly = x[-2:]
            self.goal_transform.set_translation(goalx, goaly)
            self.new_goal = False

        x = self.to_cartesian(self.q)
        theta = -self.q[0] + np.pi/2

        self.poles[0].set_rotation(theta)
        for i in range(1, self.dof+1):
            if i < self.dof:
                theta -= self.q[i]
                self.poles[i].set_translation(x[(i-1)*2], x[(i-1)*2+1])
                self.poles[i].set_rotation(theta)

            self.joints[i].set_translation(x[(i-1)*2], x[(i-1)*2+1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

