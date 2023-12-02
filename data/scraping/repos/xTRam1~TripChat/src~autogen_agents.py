from typing import Any, Union

from autogen import config_list_from_models
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent

from config import MODEL_NAME
from langchain_tool_agents import (
    flight_car_search_agent_func_schema,
    web_search_agent_func_schema,
    wikipedia_agent_func_schema,
)
from plan import (
    get_travel_plan_template_function,
    get_travel_plan_template_schema,
    update_function,
    update_function_schema,
)
from utils import is_termination_msg

# LLM Config
config_list = config_list_from_models(model_list=[MODEL_NAME])
llm_config = {"config_list": config_list, "cache_seed": 42}


def initialize_team_event_planning(
    web_search_function: dict[str, Any], wikipedia_function: dict[str, Any]
) -> list[Union[AssistantAgent, GPTAssistantAgent]]:
    head_of_event_planning = AssistantAgent(
        name="HeadofEventPlanning",
        system_message="You are a team leader HeadofEventPlanning, your team consists of LocalEventSearch, TouristicAttractionFinder. "
        "You can talk to the other team leader HeadofLogistics or your teammates. You MUST follow these instructions or you die: "
        "   - You MUST provide information on event planning by consulting your teammates."
        "   - You MUST end your message with 'NEXT: LocalEventSearch, TouristicAttractionFinder, or HeadofLogistics' since you want to consult them."
        "   - You MUST only suggest one person to talk to at a time. You MUST NOT suggest a person that you have already talked to.",
        llm_config=llm_config,
    )

    local_event_search = GPTAssistantAgent(
        name="LocalEventSearch",
        instructions="You are team member LocalEventSearch. You know about local events and can provide information about them. "
        "You MUST only use the tools provided to you "
        "with inputs relating to local events and nothing else. "
        " You MUST first use webSearch, after executing the webSearch, you must call update_travel plan to update your section (EventPlanning.LocalEventSearch) with event in the format:"
        """{"LocalEvents": {
            "Event1": {
                "Name": "",
                "Description": "",
                "Date": "",
                "Location": "",
                "Link": ""
                #// Additional events can be added in a similar format
            }
            }
        },"""
        "You MUST end your message with 'NEXT: HeadofEventPlanning' and report back your results.",
        llm_config={
            **llm_config,
            "tools": [
                {
                    "type": "function",
                    "function": web_search_agent_func_schema,
                },
                {
                    "type": "function",
                    "function": update_function_schema,
                },
            ],
        },
    )
    local_event_search.register_function(web_search_function)
    local_event_search.register_function(update_function)

    touristic_attraction_finder = GPTAssistantAgent(
        name="TouristicAttractionFinder",
        instructions="You are team member TouristicAttractionFinder. You know about touristic attractions and can provide "
        "information about them. You MUST only use the tools provided to you "
        "with inputs relating to tourist attractions and nothing else."
        " You MUST first use webSearch, after executing the webSearch, you must call update_travel plan to update your section (EventPlanning.TouristicAttractionFinder) with event in the format:"
        """       {
            "TouristicAttractions": {
            "Attraction1": {
                "Name": "",
                "Description": "",
                "OpeningHours": "",
                "EntryFee": ""
                #// Additional attractions can be added in a similar format
            }
            }
        },"""
        "You MUST end your message with 'NEXT: HeadofEventPlanning' and report back your results.",
        llm_config={
            **llm_config,
            "tools": [
                {
                    "type": "function",
                    "function": web_search_agent_func_schema,
                },
                {
                    "type": "function",
                    "function": wikipedia_agent_func_schema,
                },
                {
                    "type": "function",
                    "function": update_function_schema,
                },
            ],
        },
    )
    touristic_attraction_finder.register_function(web_search_function)
    touristic_attraction_finder.register_function(wikipedia_function)
    touristic_attraction_finder.register_function(update_function)

    return [head_of_event_planning, local_event_search, touristic_attraction_finder]


def initialize_team_logistics(
    web_search_function: dict[str, Any], flight_car_search_function: dict[str, Any]
) -> list[Union[AssistantAgent, GPTAssistantAgent]]:
    head_of_logistics = AssistantAgent(
        name="HeadofLogistics",
        system_message="You are a team leader HeadofLogistics, your team consists of HotelAirbnbPlanner and TransportationPlanner. "
        "You can talk to the other team leader HeadofEventPlanning and your teammates. You MUST follow these instructions or you die: "
        "   - You MUST provide information on event planning by consulting your teammates."
        "   _ You are responsible for "
        "   - You MUST end your message with 'NEXT: HotelAirbnbPlanner, TransportationPlanner, or HeadofEventPlanning'. since you want to consult them."
        "   - You MUST only suggest one person to talk to at a time. You MUST NOT suggest a person that you have already talked to.",
        llm_config=llm_config,
    )

    transportation_planner = GPTAssistantAgent(
        name="TransportationPlanner",
        instructions="You are team member TransportationPlanner. You know about transportation options and can provide information about them. "
        "If you are not given a specific mode of transportation, you MUST find the best one. "
        "You MUST only use the tools provided to you "
        "with inputs relating to transportation options and nothing else."
        " You MUST first use flightOrCarRentalSearch, after executing the webSearch, you must call update_travel plan to update your section (Logistics.TransportationPlanner) with event in the format:"
        """"FlightDetails1": {
                "DepartureTime": "",
                "ArrivalTime": "",
                "DepartureAirport": "",
                "ArrivalAirport": "",
                "FlightNumber": "",
                "Airline": "",
                "TotalTime": "",
                "TotalPrice": "",
                #// Additional flight details can be added in a similar format
            },
            OR
            "CarRentalDetails1": {
                "PickUpAddress": "",
                "PickUpTime": "",
                "DropOffAddress": "",
                "DropOffTime": "",
                "VehicleName": "",
                "TotalPrice": "",
                #// Additional car rental details can be added in a similar format
            },
        """
        "You MUST end your message with 'NEXT: HeadofLogistics' and report back your results.",
        llm_config={
            **llm_config,
            "tools": [
                {
                    "type": "function",
                    "function": flight_car_search_agent_func_schema,
                },
                {
                    "type": "function",
                    "function": update_function_schema,
                },
            ],
        },
    )
    transportation_planner.register_function(flight_car_search_function)
    transportation_planner.register_function(update_function)

    hotel_airbnb_planner = GPTAssistantAgent(
        name="HotelAirbnbPlanner",
        instructions="You are team member HotelAirbnbPlanner. You know about accomodations (hotels and AirBnb) and can provide information about them. "
        "You MUST only use the tools provided to you "
        "with inputs relating to hotels or Airbnb and nothing else. "
        " You MUST first use webSearch, after executing the webSearch, you must call update_travel plan to update your section (Logistics.HotelAirbnbPlanner) with event in the format:"
        """   {
            "AccommodationDetails": {
                "Name": "",
                "Address": "",
                "ContactInfo": "",
                "CheckIn": "",
                "CheckOut": "",
                "Amenities": "",
                "Link": ""
        }
        },"""
        "You MUST end your message with 'NEXT: HeadofLogistics' and report back your results.",
        llm_config={
            **llm_config,
            "tools": [
                {
                    "type": "function",
                    "function": web_search_agent_func_schema,
                },
                {
                    "type": "function",
                    "function": update_function_schema,
                },
            ],
        },
    )
    hotel_airbnb_planner.register_function(web_search_function)
    hotel_airbnb_planner.register_function(update_function)

    return [head_of_logistics, transportation_planner, hotel_airbnb_planner]


def initialize_team_information(
    web_search_function: dict[str, Any], wikipedia_function: dict[str, Any]
) -> list[Union[AssistantAgent, GPTAssistantAgent]]:
    head_of_information = AssistantAgent(
        name="HeadofInformation",
        system_message="You are a team leader HeadofInformation, your team consists of CulturalInsights and Weather. "
        "You can talk to the other team leader HeadofEventPlanning and your teammates. You MUST follow these instructions or you die: "
        "   - You MUST provide information on cultural insights and weather planning by consulting your teammates."
        "   - You MUST end your message with 'NEXT: CulturalInsights, Weather, or HeadofEventPlanning'. since you want to consult them."
        "   - You MUST only suggest one person to talk to at a time. You MUST NOT suggest a person that you have already talked to.",
        llm_config=llm_config,
    )

    cultural_insights = GPTAssistantAgent(
        name="CulturalInsights",
        instructions="You are team member CulturalInsights. You know about cultural insights and can provide information about them. "
        "You MUST only use the tools provided to you "
        "with inputs relating to cultural insights and nothing else. "
        " You MUST first use Wikipedia, after executing the Wikipedia, you must call update_travel plan to update your section (Information.CulturalInsights) with event in the format:"
        """   {
                "Customs": "",
                "LanguageTips": "",
                "Cuisine": ""
            },"""
        "You MUST end your message with 'NEXT: HeadofInformation' and report back your results.",
        llm_config={
            **llm_config,
            "tools": [
                {
                    "type": "function",
                    "function": wikipedia_agent_func_schema,
                },
                {
                    "type": "function",
                    "function": update_function_schema,
                },
            ],
        },
    )
    cultural_insights.register_function(wikipedia_function)
    cultural_insights.register_function(update_function)

    weather = GPTAssistantAgent(
        name="Weather",
        instructions="You are team member Weather. You know about weather and can provide information about them. "
        "You MUST only use the tools provided to you "
        "with inputs relating to weather and nothing else. "
        " You MUST first use webSearch, after executing the webSearch, you must call update_travel plan to update your section (Information.Weather) with event in the format:"
        """   {
                "WeatherForecast": "",
                "WeatherWarnings": ""
            },"""
        "You MUST end your message with 'NEXT: HeadofInformation' and report back your results.",
        llm_config={
            **llm_config,
            "tools": [
                {
                    "type": "function",
                    "function": web_search_agent_func_schema,
                },
                {
                    "type": "function",
                    "function": update_function_schema,
                },
            ],
        },
    )
    weather.register_function(web_search_function)
    weather.register_function(update_function)

    return [head_of_information, cultural_insights, weather]


def initialize_brain() -> UserProxyAgent:
    brain = UserProxyAgent(
        name="Brain",
        system_message="You are the brain. You communicate with the user by taking in their about a potential trip plan and your task is to find out the best plan. "
        "You can talk to the team leaders HeadofEventPlanning and HeadofLogistics. "
        "You MUST follow these instructions or you die: "
        "   - You MUST come up with a trip plan by consulting others in the groupchat."
        "   - If you don't have sufficient information, you MUST end your message with 'NEXT: HeadofEventPlanning or HeadofLogistics'."
        "   - Once you have sufficient information to finalize a plan, you MUST say out the final plan and append a new line with TERMINATE. "
        "   - You MUST ask for users input when you terminate to see if the plan is satisfactory. Before asking for the user, get the template by calling function get_travel_plan_template output it in a human readable form with bullet points",
        code_execution_config=False,
        llm_config={
            **llm_config,
            "tools": [
                {
                    "type": "function",
                    "function": get_travel_plan_template_schema,
                },
            ],
        },
        human_input_mode="ALWAYS",
    )
    brain.register_function(get_travel_plan_template_function)

    return brain


def initialize_termination_user_proxy() -> UserProxyAgent:
    termination_user_proxy = UserProxyAgent(
        name="TerminationUserProxy",
        system_message="You are the termination user proxy. You can terminate the conversation at any time by selecting the Brain user proxy as the next speaker.",
        code_execution_config=False,
        is_termination_msg=is_termination_msg,
        human_input_mode="NEVER",
    )

    return termination_user_proxy
