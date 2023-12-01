from typing import Any
import openai
import math

#ToDo:
#Early Stopping
#History and prompt curently double

#Class to handle the refinement step using the feedback in the self-refinement structure
#The Structure of this class is chosen such that it is similiar to the original self-refinement structure
class BBHTaskIterate():
    def __init__(self, model, training_prompt) -> None:
        self.model = model
        self.training_prompt = training_prompt


    async def __call__(self, problem, history, temperature=1, top_p=1, engine="gpt-3.5-turbo") -> Any:
        """
        Handle the initial request
        problem string: The initial problem from a dataset
        history string: The conversation of refinement steps
        """
        test_question = "\n Refined response:"
        prompt = "demonstrations: " + self.training_prompt + "\n" + history + test_question
        instr = """We want to iteratively improve the answer to a mathematical Problem. Use the Feedback, the conversation history and the demonstrations on your previous answer to find and fix errors in your reasoning. Create an improved answer for the question indicated with 'Question to be solved'. Make sure to write your final answer with 'the answer is' after your reasoning."""
        messages = [
        {"role": "system", "content": instr},
        {"role": "user", "content": prompt}
        ]
        retries = 0
        max_retries = 2
        error_occured = True #Makes it easier to handle, gets set to False before first call

        while (error_occured and retries<=max_retries):
            error_occured = False

            try:
                c = openai.ChatCompletion.acreate(
                                    model=engine,
                                    messages=messages,
                                    top_p = top_p,
                                    temperature = temperature
                                )
                c = await c #Waits on the response
            except openai.error.RateLimitError as e:
                print(f"Request exceeded rate limit: Error Code {e.code}")
                retries += 1
                error_occured = True
                raise e
            except openai.error.APIError as e:
                print(f"Another API error: {type(e)} with code {e.code}")
                error_occured = True
                retries += 1
            except openai.error.ServiceUnavailableError as e:
                print(f"Server Unavailable. Error Code {e.code}")
                retries += 1
                error_occured = True
            except openai.error.InvalidRequestError as e:
                print(f"Invalid Request. Probably exceeded maximum token size. Changing to larger context model Error Code {e.code}")
                retries += 1
                error_occured = True
                if engine == "gpt-3.5-turbo-16k":
                    history = history [0:500] + history[-1000:] # compress everything even tough answer might be much worse, at least an answer
                    messages = [
                        {"role": "system", "content": instr},
                        {"role": "user", "content": prompt+ "Keep your answer short."}
                    ]
                engine = "gpt-3.5-turbo-16k"
            except Exception as e:
                print(f"Got some error when Calling the engine {e}")
                retries+= 1
                error_occured = True

        if (error_occured): #Got no answer
            return history, history
        answer = c["choices"][0]["message"]["content"]#.split()#This just extracts the answer

        return answer, history + test_question + answer #Return answer and updated history
