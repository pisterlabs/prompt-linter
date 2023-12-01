import openai
import os

openai.api_key = os.environ['OPENAI_API_KEY']


class AQIInterpreter:
    def __init__(self):
        """
        This class provides an interface for interacting with the OpenAI API.

        GPT-3 is used to generate an explanation for the air quality index
        given the current, past, and future air quality index.

        """

    def get_air_quality_interpretation(self, city_name, current_aqi, past_aqi, future_aqi):
        """
        This function uses GPT-3 to generate an explanation for the air quality index
        given the current, past, and future air quality index.

        Args

        current_aqi : int
            The current air quality index.
        past_aqi : list of int
            The air quality index for the past n days.
        future_aqi : list of int
            The air quality index forecasted for next n days.

        Output
        str
            The explanation for the air quality index.
        """

        prompt = (
            f"The current air quality index for {city_name} is: {current_aqi}. "
            f"The air quality index for the past days were: {past_aqi}. "
            f"The air quality index forecasted for the next days: {future_aqi}. "
            "Provide a contextual interpretation for these values. " +
            "Include the health risks associated with the air quality index. " +
            "Don't explain the air quality index."
        )

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        return response.choices[0].text.strip()

    def interpret(self, city_name, current_aqi, past_aqi, future_aqi):
        return self.get_air_quality_interpretation(city_name, current_aqi, past_aqi, future_aqi)
