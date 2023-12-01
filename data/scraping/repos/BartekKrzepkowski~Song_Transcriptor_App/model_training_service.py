from training_data import defaultPrompt
import openai


def set_openai_key(key):
    """Sets OpenAI key."""
    openai.api_key = key


class Code:
    def __init__(self):
        print("Model Intilization--->")

    def query(self, completion_kwargs, topic):
        """
        wrapper for the API to save the prompt and the result
        """

        # arguments to send the API
        kwargs = {}
        prompt = defaultPrompt[topic].format(*completion_kwargs[topic])
        for kwarg in completion_kwargs:
            if kwarg not in ['api_key', topic]:
                kwargs[kwarg] = completion_kwargs[kwarg]

        r = openai.Completion.create(prompt=prompt, **kwargs)["choices"][0]["text"].strip()
        return r

    def model_prediction(self, completion_kwargs, topic):
        """
        wrapper for the API to save the prompt and the result
        """
        # Setting the OpenAI API key got from the OpenAI dashboard
        set_openai_key(completion_kwargs['api_key'])
        output = self.query(completion_kwargs, topic)
        return output
