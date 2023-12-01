import openai
class GoalManager:
    def __init__(self, gpt, emotion):
        # ...
        self.current_goal = None
        self.gpt = gpt
        self.emotion = emotion
    
    def decide_goal(self, thought):
        systemprompt = """I am a Cognitive Entity.
        I need to set a goal or goals based on my thoughts."""
        prompt = """{thought}"""  # you may want to customize this prompt
        
        chat_input = prompt.format(thought=thought)
        chat_output = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
        return chat_output.choices[0].message['content']
        gpt_response = self.gpt.process_thought(systemprompt, prompt.format(thought=thought))
        # logic to decide and set the current goal based on the thought
        goal = gpt_response 
        self.set_goal(goal)
        return goal
    
    def set_goal(self, goal):
        self.current_goal = goal

    def get_goal(self, thought):
        systemprompt = """I am a Cognitive Entity.
        I need to set a goal or goals based on my thoughts.
        Only reply with a goal."""
        prompt = """My current thought is:
        {thought}"""
        result = self.gpt.process_thought(systemprompt, prompt.format(thought=thought))
        print(result)
        return result

    def reset_goal(self):
        self.current_goal = None
