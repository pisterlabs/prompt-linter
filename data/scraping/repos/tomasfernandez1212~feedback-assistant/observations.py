from typing import List
import openai
from src.llm.utils import unpack_function_call_arguments
from src.data.observations import Observation


def generate_observations(text: str) -> list[Observation]:
    response = openai.ChatCompletion.create(  # type: ignore
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert in customer service. Your task is to interpret customer's reviews, feedback, and conversations with us to infer facts about their experience. 
                
                For example, if a customer says "A fun Greek spot for a quick bite -- located right on Divisadero, it looks a bit hole-in-the-wall but has a lovely back patio with heaters. You order at the front and they give you a number, so it's a fast-casual vibe. The staff are very sweet and helpful, too. There aren't large tables, so would recommend going with a smaller group. The food is fresh and healthy, with a selection of salads and gyros! The real star of the show is dessert. The gryo is a little hard to eat IMHO -- I got the white sweet potato and ended up just using a fork and knife, and I didn't think the flavor was anything too memorable. The fries are SO good -- crispy on the outside, soft on the inside. My absolute favorite thing was the Baklava Crumbles on the frozen Greek yoghurt -- literally... FIRE. The yoghurt is tart and the baklava is sweet and I am obsessed. I'd come back for that alone.",

                You would identify: ["The customer visited our location on Divisadero.", "The exterior of the restaurant may seem modest or unassuming, as described as 'hole-in-the-wall'.", "The restaurant has a back patio equipped with heaters.", "The ordering system is more casual, with customers placing orders at the front and given a number to wait for their food.", "The staff of the restaurant made a positive impression on the customer, being described as 'sweet' and 'helpful'.", "The restaurant is not suitable for larger groups due to the lack of large tables.", "The food offered is fresh and healthy, including options like salads and gyros.", "The customer found the gyro hard to eat and not particularly flavorful, specifically mentioning a white sweet potato gyro.", "The restaurant serves high-quality fries that are crispy on the outside and soft on the inside.", "The restaurant offers a dessert option that involves Baklava Crumbles on frozen Greek yogurt.", "The customer was highly impressed with the Baklava Crumbles on frozen Greek yogurt, describing it as 'FIRE' and expressing an eagerness to revisit the restaurant for this dessert.", "The customer found the combination of tart yogurt and sweet baklava to be very satisfying."]

                The goal is to infer observations from customers' experiences.""",
            },
            {
                "role": "user",
                "content": f"What can we infer here: {text}",
            },
        ],
        functions=[
            {
                "name": "report_interpretation",
                "description": "Used to report interpretations to the system.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "observations": {
                            "type": "array",
                            "description": "A list of interpretations.",
                            "items": {"type": "string"},
                        },
                    },
                },
                "required": ["observations"],
            }
        ],
        function_call={"name": "report_interpretation"},
    )

    list_of_observation_texts: List[str] = unpack_function_call_arguments(response)["observations"]  # type: ignore
    list_of_observations: List[Observation] = []
    for observation_text in list_of_observation_texts:
        list_of_observations.append(Observation(text=observation_text))
    return list_of_observations
