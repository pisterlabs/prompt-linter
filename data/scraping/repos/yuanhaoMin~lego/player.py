from agent.action_agent import ActionAgent
from agent.critic_agent import CriticAgent
from agent.skill_agent import SkillAgent
from langchain.schema import AIMessage, BaseMessage
from time import time
from util.ansi_console_utils import print_action_agent_ai_message, print_error_message


class Player:
    def __init__(self, resume: bool):
        self.action_agent = ActionAgent(
            model_name="gpt-4",
            request_timout=120,
            show_execution_error=True,
        )
        self.critic_agent = CriticAgent(
            model_name="gpt-3.5-turbo",
            request_timout=60,
        )
        self.skill_agent = SkillAgent(
            model_name="gpt-3.5-turbo",
            request_timout=60,
            resume=resume,
            retrieval_top_k=5,
        )
        self.action_agent_rollout_num_iter = -1
        self.action_agent_task_max_retries = 4
        self.messages = None
        self.task = None

    def reset(self, task: str) -> list[BaseMessage]:
        self.action_agent_rollout_num_iter = 0
        self.task = task
        skills = self.skill_agent.retrieve_skills(query=self.task)
        system_message = self.action_agent.render_system_message(skills=skills)
        human_message = self.action_agent.render_human_message(
            code="", error_message="", output="", critique="", task=self.task
        )
        self.messages = [system_message, human_message]
        return self.messages

    def step(self) -> tuple[list[BaseMessage], bool]:
        if self.action_agent_rollout_num_iter < 0:
            raise ValueError("Agent must be reset before stepping")
        error_message = ""
        ai_message = self.action_agent.llm(self.messages)
        print_action_agent_ai_message(message_content=ai_message.content)
        program_call, program_code, program_name = self.action_agent.process_ai_message(
            ai_message=ai_message
        )
        # Code must before call
        code = program_code + "\n" + program_call
        error_message, output = self.action_agent.execute_code_and_gather_info(
            code=code
        )
        success = False
        critique = ""
        critique, success = self.critic_agent.check_task_success(
            error_message=error_message,
            output=output,
            task=self.task,
            max_retries=5,
        )
        if not success:
            # TODO revert actions (db operations) since execution failed
            skills = self.skill_agent.retrieve_skills(query=self.task)
            system_message = self.action_agent.render_system_message(skills=skills)
            human_message = self.action_agent.render_human_message(
                code=program_code,
                error_message=error_message,
                output=output,
                critique=critique,
                task=self.task,
            )
            self.messages = [system_message, human_message]
        self.action_agent_rollout_num_iter += 1
        done = (
            self.action_agent_rollout_num_iter >= self.action_agent_task_max_retries
            or success
        )
        info = {
            "program_call": program_call,
            "program_code": program_code,
            "program_name": program_name,
            "success": success,
        }
        return self.messages, done, info

    def rollout(self, task) -> tuple[list[BaseMessage], bool]:
        self.reset(task=task)
        while True:
            messages, done, info = self.step()
            if done:
                break
        return messages, done, info

    def learn(self) -> None:
        try:
            start_time = time()
            messages, done, info = self.rollout(
                "获得轴承有关的订单, 增加300的成本并更新订单. 最后计算出该订单的总成本(数量乘以成本)"
            )
            end_time = time()
        except Exception as e:
            done = False
            error_message = f"Rollout出错: {e}, 判断任务失败"
            print_error_message(message_content=error_message)
        if info["success"]:
            self.skill_agent.add_new_skill(info)
            print(f"Rollout共耗时{end_time - start_time}s")


player = Player(resume=True)
player.learn()
