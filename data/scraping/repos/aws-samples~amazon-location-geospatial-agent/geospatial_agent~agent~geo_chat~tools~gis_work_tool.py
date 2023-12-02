from langchain.tools import Tool

from geospatial_agent.agent.action_summarizer.action_summarizer import ActionSummarizer
from geospatial_agent.agent.geospatial.agent import GeospatialAgent

GIS_WORK_TOOL = "gis_work_tool"


def gis_work_tool(session_id: str, storage_mode: str, action_summarizer=None, gis_agent=None):
    desc = f"""\
A tool that invokes a {GeospatialAgent.__name__} if the user action is requires geospatial analysis to be done on user provided data.
{GeospatialAgent.__name__} description: {GeospatialAgent.__doc__}

It accepts two inputs: user_input and session_id.

An example query might look like the following:
Draw time series choropleth map of weather temperature change over major cities of the world.

Data Locations:
1. Climate Change: Earth Surface Temperature Data location since 1940s data location: GlobalLandTemperaturesByCity.csv

A qualified action for the tool have the following requirements:
1. A geospatial analysis action such as heatmap, choropleth, or time series.
2. A data location such as a scheme://URI or just a file name such as data.csv.

DO NOT invoke this tool unless both of these requirements are met.

This tool will invoke the GIS agent to perform the geospatial analysis on the data.
The return is freeform string or a URL to the result of the analysis."""

    if action_summarizer is None:
        action_summarizer = ActionSummarizer()

    if gis_agent is None:
        gis_agent = GeospatialAgent(storage_mode=storage_mode)

    def gis_work_tool_func(user_input: str):
        action_summary = action_summarizer.invoke(
            user_input=user_input, session_id=session_id, storage_mode=storage_mode)
        output = gis_agent.invoke(action_summary=action_summary, session_id=session_id)

        return "Observation: GIS Agent has completed it's work. This is the final answer."

    return Tool.from_function(func=gis_work_tool_func, name=GIS_WORK_TOOL, description=desc)
