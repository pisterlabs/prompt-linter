from typing import List
import openai
from src.data.observations import Observation
from src.data.actionItems import ActionItem
from src.llm.scores import Score, ScoreNames

from src.llm.utils import unpack_function_call_arguments

# Used to filter observations that indicate satisfaction, lack specificity, or have no impact on business outcomes.
ACTION_THRESHOLDS = {
    ScoreNames.SATISFACTION: {
        "min": 0,
        "max": 60,
    },
    ScoreNames.SPECIFICITY: {
        "min": 30,
        "max": 100,
    },
    ScoreNames.BUSINESS_IMPACT: {
        "min": 10,
        "max": 100,
    },
}


def check_needs_action(scores: List[Score]) -> bool:
    """
    Given a list of scores for an observation, check if the scores meet the thresholds for action items.
    """
    for score in scores:
        if score.name in ACTION_THRESHOLDS:
            if score.score < ACTION_THRESHOLDS[score.name]["min"]:
                return False
            if score.score > ACTION_THRESHOLDS[score.name]["max"]:
                return False
    return True


def generate_action_items(
    feedback_item: str,
    observations: List[Observation],
    existing_action_items: List[ActionItem],
) -> List[ActionItem]:
    """
    Given a feedback item, a list of observations requiring actions, and a list of existing action items, return a list of new action items to add.
    """

    if len(observations) == 0:
        return []

    numbered_observations = ""
    for i, observation in enumerate(observations):
        numbered_observations += f"{i}. {observation.text}\n"

    # Create a numbered list of existing action items
    if len(existing_action_items) == 0:
        numbered_existing_action_items = "[]"
    else:
        numbered_existing_action_items = ""
        for i, action_item_text in enumerate(existing_action_items):
            numbered_existing_action_items += f"{i}. {action_item_text.text}\n"

    response = openai.ChatCompletion.create(  # type: ignore
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in customer service. Your task is to interpret customer feedback to infer action items we can take to improve customer experience. ",
            },
            {
                "role": "user",
                "content": f"FEEDBACK FROM CUSTOMER:\n\n{feedback_item}\n\n---\nFROM THIS FEEDBACK, WE HAVE MADE THE FOLLOWING OBSERVATIONS WHICH MIGHT REQUIRE ACTION TO BE TAKEN:\n\n{numbered_observations}\n\n---\nHERE ARE THE EXISTING ACTION ITEMS WE HAVE IN OUR BACKLOG:\n\n{numbered_existing_action_items}\n\n---\nWHAT ACTION ITEMS DO WE NEED TO ADD TO OUR BACKLOG FOR THOSE OBSERVATIONS IF ANY? DON'T ADD ACTION ITEMS IF THE ONES IN THE BACKLOG ALREADY ADDRESS THE ISSUE.",
            },
        ],
        functions=[
            {
                "name": "report_action_items",
                "description": "This function is used to add more action items.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_items": {
                            "type": "array",
                            "description": "A list of action items. For example: ['Evaluate the presentation of the gyro dish, particularly the white sweet potato gyro, to make it easier to eat. The customer found it hard to consume in its current form.', 'Maintain the quality of the fries, as they received high praise from the customer.']",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["action_items"],
                },
            },
        ],
        function_call={"name": "report_action_items"},
    )

    new_action_items_text: List[str] = unpack_function_call_arguments(response)["action_items"]  # type: ignore

    new_action_items: List[ActionItem] = []
    for action_item_text in new_action_items_text:
        action_item = ActionItem(text=action_item_text)
        new_action_items.append(action_item)

    return new_action_items  # type: ignore
