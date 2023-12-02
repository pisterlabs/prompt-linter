import time
import openai

def robust_api_call(call, retries=3, delay=2, rate_limit_delay=10):
    for _ in range(retries):  # Attempt the API call n times
        try:
            return call()  # Perform the API call and return the result if successful
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}. Retrying...")
            time.sleep(delay)  # Wait for a specified delay before retrying
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}. Retrying...")
            time.sleep(delay)
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}. Retrying after a longer delay...")
            time.sleep(rate_limit_delay)  # Wait longer if rate limit has been exceeded
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API service unavailable: {e}. Retrying...")
            time.sleep(rate_limit_delay)  # Wait for a specified delay before retrying
    return None  # Return None if the API call failed after all retries