import openai
import time
from info import api_key


# This class is responsible for reflecting on the agent's recent behavior and updating its traits accordingly
class Reflection:
    def run(self, agent):
        """
        Reflects on the agent's recent behavior and updates its traits accordingly.
        """
        try:
            relevant_memories = agent.get_relevant_memories("recent behavior")
            for memory in relevant_memories:
                for trait in agent.traits:
                    assert hasattr(agent.traits[trait], "__add__") and hasattr(agent.traits[trait], "__sub__"), "Traits should be of a type that supports addition and subtraction."
                    if trait in memory.content:
                        if "positive" in memory.content:
                            agent.traits[trait] += 1
                        elif "negative" in memory.content:
                            agent.traits[trait] -= 1
        except Exception as e:
            print(f"Failed to reflect: {e}")


# This class is responsible for planning the agent's next action based on its goals
class Planning:
    def run(self, agent):
        """
        Plans the agent's next action based on its goals.
        """
        try:
            for goal in agent.goals:
                assert isinstance(goal, dict) and "priority" in goal and "action" in goal, "Goals should be dictionaries with 'priority' and 'action' keys."
                if goal["priority"] > 5:
                    agent.plan = goal["action"]
                    break
            else:
                agent.plan = "continue current activity"
        except Exception as e:
            print(f"Failed to plan: {e}")


# This class is responsible for reacting to stimuli and updating the agent's behavior accordingly
class Reaction:
    def run(self, agent):
        """
        Updates the agent's behavior based on stimuli.
        """
        try:
            for stimulus in agent.stimuli:
                assert isinstance(stimulus, dict) and "intensity" in stimulus and "response" in stimulus, "Stimuli should be dictionaries with 'intensity' and 'response' keys."
                if stimulus["intensity"] > 5:
                    agent.behavior = stimulus["response"]
                    break
            else:
                agent.behavior = "continue current behavior"
        except Exception as e:
            print(f"Failed to react: {e}")


# This class is responsible for generating responses to interview questions
class Interview:
    def __init__(self, agent):
        # assert hasattr(agent.language_model, "generate"), "Language model should have a 'generate' method."
        self.language_model = agent.language_model


    def generate_response(self, query):
        """
        Generates a response to a query using a language model.
        """
        try:
            prompt = f"Q: {query}\nA:"
            response = self.language_model.generate(prompt)
            return response
        except Exception as e:
            print(f"Failed to generate response: {e}")


# This class is responsible for summarizing the agent's characteristics, occupation, and feelings
class Summarization:
    def __init__(self, agent):
            self.summary = {}
            # self.api_key = api_key
            # openai.api_key = api_key
            self.last_updated = time.time()
            self.language_model = agent.language_model

    def needs_update(self, agent):
        """
        Checks if the summarization needs to be updated.
        """
        latest_memory = agent.memory.get_latest_memory()  # Assumes a method that retrieves the latest memory
        if latest_memory is None or latest_memory['time'] > self.last_updated:
            return True
        return False

    def run_characteristics(self, agent):
        """
        Summarizes the agent's personality traits.
        """
        relevant_memories = agent.get_relevant_memories("personality traits")
        trait_counts = {}
        for memory in relevant_memories:
            for trait in agent.traits:
                if trait in memory.content:
                    if trait not in trait_counts:
                        trait_counts[trait] = 0
                    trait_counts[trait] += 1
        self.summary["personality_traits"] = trait_counts

    def run_occupation(self, agent):
        """
        Summarizes the agent's occupation.
        """
        relevant_memories = agent.get_relevant_memories("occupation")
        if relevant_memories:
            self.summary["occupation"] = relevant_memories[-1].content  # The occupation from the latest memory

    def run_feeling(self, agent):
        """
        Summarizes the agent's feelings.
        """
        relevant_memories = agent.get_relevant_memories("feeling")
        if relevant_memories:
            self.summary["feelings"] = relevant_memories[-1].content  # The feelings from the latest memory
    
    def combine_outputs(self):
        """
        Combines the outputs of the summarization.
        """
        combined_summary = "Traits: {}\nOccupation: {}\nFeelings: {}".format(
            self.summary.get("personality_traits", "N/A"),
            self.summary.get("occupation", "N/A"),
            self.summary.get("feelings", "N/A")
        )
        self.summary["combined"] = combined_summary
        self.last_updated = time.time()  # Updating the last updated time after combining the outputs
    
    def generate_response(self, query):
        """
        Generates a response to a given query using a language model.
        """
        prompt = f"Q: {query}\nA:"
        try:
            response = self.language_model.generate("Q: {query}\nA:")
            # response = openai.Completion.create(
            #   engine="text-davinci-002",
            #   prompt=prompt,
            #   max_tokens=1024,
            #   n=1,
            #   stop=None,
            #   temperature=0.5,
            # )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Failed to generate response: {e}")
            return None

    def run(self, agent):
        """
        Runs the summarization.
        """
        if self.needs_update(agent):
            self.run_characteristics(agent)
            self.run_occupation(agent)
            self.run_feeling(agent)
        self.combine_outputs()



