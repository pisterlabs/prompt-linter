from openai_decorator import openaifunc, get_openai_funcs

@openaifunc
def add_numbers(a: int, b: int):
    """
    This function adds two numbers.
    """
    return a + b

print(get_openai_funcs())
