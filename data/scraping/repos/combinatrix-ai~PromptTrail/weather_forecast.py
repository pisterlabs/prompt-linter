# This example shows how to create a simple agent for function calling.

import enum
import os
from typing import Any, Dict, Optional, Sequence

from prompttrail.agent import State
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import LinearTemplate, MessageTemplate
from prompttrail.agent.templates.openai import OpenAIGenerateWithFunctionCallingTemplate
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

# First, we must define the IO of the function.

# The function takes two arguments: place and temperature_unit.
# The function returns the weather and temperature.


# Start with the arguments.
# We define the arguments as a subclass of ToolArgument.
# value is the value of the argument. Define the type of value here.
class Place(ToolArgument):
    description: str = "The location to get the weather forecast"
    value: str


# If you want to use enum, first define the enum.
class TemperatureUnitEnum(enum.Enum):
    Celsius = "Celsius"
    Fahrenheit = "Fahrenheit"


# And then you can use the class as the type of value.
# Note that if you set the type as Optional, it means that the argument is not required.
class TemperatureUnit(ToolArgument):
    description: str = "The unit of temperature"
    value: Optional[TemperatureUnitEnum]


# We can instantiate the arguments like this:
# place = Place(value="Tokyo")
# temperature_unit = TemperatureUnit(value=TemperatureUnitEnum.Celsius)
# Howwever, this is the job of the function itself, so we don't need to do this here.


# Next, we define the result.
# We define the result as a subclass of ToolResult.
# The result must have a show method that can pass the result to the model.
class WeatherForecastResult(ToolResult):
    temperature: int
    weather: str

    def show(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "weather": self.weather}


# Finally, we define the function itself.
# The function must implement the _call method.
# The _call method takes a list of ToolArgument and returns a ToolResult.
# Passed arguments are compared with argument_types and validated. This is why we have to define the type of arguments.
class WeatherForecastTool(Tool):
    name = "get_weather_forecast"
    description = "Get the current weather in a given location and date"
    argument_types = [Place, TemperatureUnit]
    result_type = WeatherForecastResult

    def _call(self, args: Sequence[ToolArgument], state: State) -> ToolResult:
        return WeatherForecastResult(temperature=0, weather="sunny")


# Let's define a template that uses the function.
template = LinearTemplate(
    templates=[
        MessageTemplate(
            role="system",
            content="You're an AI weather forecast assistant that help your users to find the weather forecast.",
        ),
        MessageTemplate(
            role="user",
            content="What's the weather in Tokyo tomorrow?",
        ),
        # In this template, two API calls are made.
        # First, the API is called with the description of the function, which is generated automatically according to the type definition we made.
        # The API return how they want to call the function.
        # Then, according to the response, runner call the function with the arguments provided by the API.
        # Second, the API is called with the result of the function.
        # Finally, the API return the response.
        # Therefore, this template yields three messages. (sender: assistant, function, assistant)
        OpenAIGenerateWithFunctionCallingTemplate(
            role="assistant",
            functions=[WeatherForecastTool()],
        ),
    ]
)

runner = CommandLineRunner(
    model=OpenAIChatCompletionModel(
        configuration=OpenAIModelConfiguration(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    ),
    parameters=OpenAIModelParameters(
        model_name="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0,
    ),
    template=template,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

if __name__ == "__main__":
    runner.run()
