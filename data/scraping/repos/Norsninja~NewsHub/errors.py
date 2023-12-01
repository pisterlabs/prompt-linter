import time
import openai

def robust_api_call(call, retries=3, base_delay=2):
    delay = base_delay
    for attempt in range(retries):
        try:
            print(f"Attempting API call, attempt {attempt + 1}")
            return call()
        except openai.error.APIError as e:
            print(f"API Error: {e}. Attempt {attempt + 1} of {retries}")
        except openai.error.APIConnectionError as e:
            print(f"Connection Error: {e}. Attempt {attempt + 1} of {retries}")
        except openai.error.RateLimitError as e:
            print(f"Rate Limit Exceeded: {e}. Waiting {delay} seconds.")
        except openai.error.ServiceUnavailableError as e:
            print(f"Service Unavailable: {e}. Waiting {delay} seconds.")
        except openai.error.Timeout as e:
            print(f"Timeout: {e}. Waiting {delay} seconds.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Attempt {attempt + 1} of {retries}")
        time.sleep(delay)
        delay *= 2  # Exponential backoff for any retry scenario
    print("All API call attempts failed.")
    return None
