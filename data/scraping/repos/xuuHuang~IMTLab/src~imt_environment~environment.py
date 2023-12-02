import time
import logging
import openai

logger = logging.getLogger("environment")

class Environment():
    def __init__(self, imt_system, policy) -> None:
        self.imt_system = imt_system
        self.policy = policy
        self.state = State()

    def initialize_episode(self, src, tgt):
        self.state.initialize_episode()
        self.policy.initialize_episode()
        self.episode_over = False
        self.src = src.strip()
        self.tgt = tgt.strip()
        self.template = None
        self.hypo = None
        self.tolerance = 3

    def next_turn(self):
        if self.state.turn == 0:
            start_time = time.time()
            try:
                self.hypo = self.imt_system.translate(self.src)
                response_time = time.time() - start_time
            except (openai.error.APIError, openai.error.RateLimitError, openai.error.APIConnectionError) as e:
                print(e)
                self.tolerance -= 1
                if self.tolerance < 0:
                    print("exit")
                    exit(1)
                time.sleep(5)
                return False, None
            self.max_turn = self.policy.max_turn(self.hypo, self.tgt)
            self.tolerance = 3
        else:
            if self.tolerance == 3:
                self.template, editing_cost, failed = self.policy.revise(self.hypo, self.tgt)
                if failed:
                    logger.info("policy failed!")
                    return self.end_episode(False)
                self.state.editing_cost += editing_cost
                hypo_tmp = self.template.template2hypo()
                if self.policy.accept(hypo_tmp, self.tgt):
                    logger.info("accept at turn {}!".format(self.state.turn))
                    return self.end_episode(True)
            start_time = time.time()
            try:
                hypo = self.imt_system.translate(self.src, self.template)
                response_time = time.time() - start_time
            except (openai.error.APIError, openai.error.RateLimitError, openai.error.APIConnectionError) as e:
                print(e)
                self.tolerance -= 1
                if self.tolerance < 0:
                    print("exit")
                    exit(1)
                time.sleep(5)
                return False, None
            self.tolerance = 3
            if hypo == self.hypo:
                logger.warning("same hypothesis with the last turn!")
            else:
                self.state.consistency += self.policy.consistency(self.hypo, hypo)
            self.hypo = hypo

        self.state.response_time += response_time
        self.state.turn += 1

        if self.policy.accept(self.hypo, self.tgt):
            logger.info("accept at turn {}!".format(self.state.turn))
            return self.end_episode(True)
            
        if self.state.turn >= self.max_turn:
            logger.info("episode failed!")
            return self.end_episode(False)

        return self.episode_over, self.state.get_state()

    def end_episode(self, success):
        self.episode_over = True
        if success:
            self.state.success = True
            self.imt_system.update(self.src, self.hypo)
        else:
            _, _, editing_cost = self.policy.post_editing(self.hypo, self.tgt)
            self.state.editing_cost += editing_cost
        self.state.norm_editing_cost = self.state.editing_cost / len(self.tgt)
        return self.episode_over, self.state.get_state()

class State():
    def initialize_episode(self):
        self.turn = 0
        self.response_time = 0
        self.editing_cost = 0
        self.norm_editing_cost = 0
        self.success = False
        self.consistency = 0

    def get_state(self):
        return {
            "turn": self.turn,
            "editing_cost": self.editing_cost,
            "normalized_editing_cost": self.norm_editing_cost,
            "success": self.success,
            "response_time": self.response_time,
            "consistency": self.consistency,
        }
