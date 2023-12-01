import openai_api

class SolutionProposer:
    def __init__(self):
        self.generated_result = ""

    def propose_solutions(self, summarized_content):
        # Use the OpenAI API to generate brief or detailed orientation axes
        self.generated_result = openai_api.generate_orientation_axes(summarized_content)
