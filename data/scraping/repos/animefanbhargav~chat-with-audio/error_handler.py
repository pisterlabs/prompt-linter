
import logging
from typing import Callable, Any, Union
from openai.error import AuthenticationError, APIConnectionError


def openai_error_handler(func: Callable, func_args: Any) -> dict[str, Union[str, bool]]:
    """
    custom callback for handling open ai errors

    Args:
        func: the function to exceute within try block
        func_args: arguments to pass to the function when calling
    Returns:
        result: dictionary containing the result or error as 'result' key's value and 
                'error_message' is True or False depending on whether an error occured.
    """
    error_occured = True
    try:
        result = func(func_args)
        error_occured = False
    except TypeError as e:
        result = f':red[{e}]'
    except ConnectionError:
        result = ':red[Failed to Connect]'
    except ValueError as e:
        logging.exception(e)
        result = f':red[{e}]'
    except AuthenticationError as e:
        logging.exception(e)
        result = f':red[Invalid API Key]'
    except APIConnectionError as e:
        logging.exception(e)
        result = f':red[Error communicating with OpenAI]'
    except Exception as e:
        logging.exception(e)
        result = f':red[{e}]'
    finally:
        return {'result': result, "error_occured": error_occured}
