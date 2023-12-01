"""
Weather Functions:
A collection of functions to provide current weather information and forecasts.
"""

from openai_decorator import openaifunc

@openaifunc
def curr_weather(location: str) -> str:
    """
    Gets the current weather information for the given location.
    :param location: The location for which to get the weather
    """
    return "The weather is nice and sunny in " + location

@openaifunc
def tomorrow_weather(location: str) -> str:
    """
    Gets the weather forecast for tomorrow for the given location.
    :param location: The location for which to get the forecast
    """
    return "Tomorrow's weather will be rainy in " + location
