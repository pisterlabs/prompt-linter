"""This file is a copy of multi_monitering.py from OpenAI MultiAgent compitition repo
Source:
https://github.com/openai/multiagent-competition.git"""

from gym.wrappers import Monitor

class MultiMonitor(Monitor):

    def _before_step(self, action):
        return

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if done[0] and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self._reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record video
        self.video_recorder.capture_frame()

        return done
