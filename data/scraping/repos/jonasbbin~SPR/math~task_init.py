from typing import Any
import openai
import math
import async_timeout

#Class to handle the intial in the self-refinement structure
#The Structure of this class is chosen such that it is similiar to the original self-refinement structure
class MathTaskInit():
    def __init__(self, model, training_prompt):
        self.model = model
        self.training_prompt = training_prompt


    async def __call__(self, problem, instruction = "blablba", temperature=1, top_p=1) -> Any:
        """
        Handle the initial request
        problem string: The initial problem from a dataset
        """
        test_question = "\n 'Question to be solved':" + problem
        prompt = self.training_prompt + test_question
        if instruction == "blablba":
            messages = [
            {"role": "system", "content": "Follow the given examples and answer the question."},
            {"role": "user", "content": prompt}
            ]
        else:
            prompt = self.training_prompt + "\n"+instruction +"\n"+ test_question
            messages = [
            {"role": "user", "content": prompt}
            ]
            
        retries = 0
        max_retries = 2
        error_occured = True #Makes it easier to handle, gets set to False before first call

        while (error_occured and retries<=max_retries):
            error_occured = False
            
            try:
                async with async_timeout.timeout(60):    
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
            except Exception as e:
                print(f"Got some error when Calling the engine {e}, Amount of retries {retries}")
                retries+= 1
                error_occured = True

        if (error_occured): #Got no answer
            return 0, test_question
        answer = c["choices"][0]["message"]["content"]#This just extracts the answer
        
        return answer, test_question + "\n" +answer
    
    
