from typing import Any
import openai
import math

#Class to handle the feedback step using the previous answer in the self-refinement structure
#The Structure of this class is chosen such that it is similiar to the original self-refinement structure
class GSMFeedback():
    def __init__(self, model, training_prompt) -> None:
        self.model = model
        self.training_prompt = training_prompt

    async def __call__(self, problem, history, single=False, temperature=1, top_p=1) -> Any:
        """
        Handle the initial request
        problem string: The initial problem from a dataset
        history string: The conversation of refinement steps
        single: Indicates if there is an iterative step
        """
        prompt = history
        if single:
            instruction = "There is an error in the response above because of lack of understanding of the question. What is the error? To find the error, go through the answer step-by-step and fix it. If there is no error write 'there is no error' together with the solution with 'the answer is '."
        else:
            instruction = "There is probably an error in the response above because of lack of understanding of the question. What is the error? Give Feedback to the proposed solution. Make sure to think step-by-step."
        messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
        ]
        retries = 0
        max_retries = 2
        error_occured = True #Makes it easier to handle, gets set to False before first call

        while (error_occured and retries<=max_retries):
            error_occured = False

            try:
                c = openai.ChatCompletion.acreate(
                                    model=self.model,
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
                engine = "gpt-3.5-turbo-16k"
            except Exception as e:
                print(f"Got some error when Calling the engine {e}")
                retries+= 1
                error_occured = True

        if (error_occured): #Got no answer
            return None, ""
        answer = c["choices"][0]["message"]["content"]#This just extracts the answer
        
        solved = False
        if "it is correct" in answer.lower() or "there is no error" in answer.lower() or "response is correct" in answer.lower() or "solution is correct" in answer.lower():
            solved = True
        
        if single:
            history = history +"\n Next Response: " + answer
        else:
            history = history +"\n Feedback: " + answer

        return answer, history, solved #Answer as well as history
