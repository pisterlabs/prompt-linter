import openai

class ZeroShotReact:
    def __init__(self, tools):
        self.tools = tools

    def perform_action(self, prompt):
        # Generate a list of descriptions from the tools
        descriptions = [tool['description'] for tool in self.tools]

        # Use the OpenAI API to generate a list of scores for each description
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt + "\n" + "\n".join(descriptions),
            temperature=0.0,
            max_tokens=1
        )

        # Find the tool with the highest score
        best_tool = self.tools[response['choices'][0]['text'].strip()]

        # Execute the best tool and return its result
        return best_tool['function']()
