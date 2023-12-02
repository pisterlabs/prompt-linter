import requests
import aiohttp
from requests.exceptions import HTTPError
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Type


class ForecastInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")

class ForecastOutput(BaseModel):
    temperature: Optional[float] = Field(None, description="Temperature in degrees Celsius")
    wind_speed: Optional[float] = Field(None, description="Wind speed in meters per second")
    precipitation: Optional[float] = Field(None, description="Precipitation in millimeters")

class ForecastTool(StructuredTool):
    name: str = "GetWeatherForecast"
    description: str = "Useful when you need to answer a question about weather in a specific location."
    args_schema: Type[BaseModel] = ForecastInput
        # SMHI API endpoint for weather forecast
    smhi_url = "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/{input.longitude}/lat/{input.latitude}/data.json"


    def _run(self, latitude: float, longitude: float) -> ForecastOutput:
        print(f"(sync) Retrieving weather forecast for lat: {latitude}, lon: {longitude}")

        url = self.smhi_url.format(input=ForecastInput(latitude=latitude, longitude=longitude))
        response = requests.get(url)

        if response.status_code == 200:
            forecast=response.json()
            return self.extract_weather_info(forecast=forecast)
        else:
            raise HTTPError(f'Unexpected status code: {response.status_code}')
        
    async def _arun(self, latitude: float, longitude: float) -> ForecastOutput:
        print(f"(async) Retrieving weather forecast for lat: {latitude}, lon: {longitude}")

        url = self.smhi_url.format(input=ForecastInput(latitude=latitude, longitude=longitude))
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    forecast = await response.json()
                    return self.extract_weather_info(forecast=forecast)
                else:
                    raise HTTPError(f'Unexpected status code: {response.status_code}')

    def extract_weather_info(self, forecast: dict) -> ForecastOutput:                
        if 'timeSeries' in forecast and len(forecast['timeSeries']) > 0:
            # The first element in the time series is usually the current weather forecast
            current_forecast = forecast['timeSeries'][0]

            weather_info: ForecastOutput = {
                'temperature': None,
                'wind_speed': None,
                'precipitation': None
            }

            for parameter in current_forecast['parameters']:
                if parameter['name'] == 't':  # Temperature
                    weather_info['temperature'] = parameter['values'][0]
                elif parameter['name'] == 'ws':  # Wind speed
                    weather_info['wind_speed'] = parameter['values'][0]
                elif parameter['name'] == 'pmean':  # Mean precipitation
                    weather_info['precipitation'] = parameter['values'][0]

            return weather_info
        else:
            raise KeyError("Error: Could not parse the weather forecast.")