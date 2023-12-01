from loguru import logger
from ai_driver.flows.swarm import SwarmConfig, LLMSwarm
from ai_driver.cloud_llm.OAISettings import OAIModels
from ai_driver.cloud_llm.open_ai_api import OpenAIApiCaller


def get_swarm_output(
    base_chat_history,
    iteration_history,
    instruction,
    system_messages,
    temperature,
    token_length,
    feedback_rounds,
    model_for_system_messages,
    model_for_answer_generation,
    model_for_answer_iteration,
    model_for_synthesizing,
):
    # Log the configuration settings
    config = SwarmConfig(
        num_agents=system_messages,
        max_response_tokens=token_length,
        iterations=feedback_rounds,
        system_message_model=OAIModels(model_for_system_messages),
        answer_generation_model=OAIModels(model_for_answer_generation),
        answer_iteration_model=OAIModels(model_for_answer_iteration),
        response_aggregation_model=OAIModels(model_for_synthesizing),
        temperature=temperature,
    )

    logger.info(config)
    swarm = LLMSwarm(instruction, config=config, APICaller=OpenAIApiCaller)
    # exit()
    synthesized_response, iterations = swarm.generate_response()

    # Appending iterations to iteration_history
    for iter_num, iter_data in iterations.items():
        for k, v in iter_data.items():
            formatted_message = f"Iteration {iter_num} - {k}: {v}"
            iteration_history.append(("", formatted_message.replace("\n", "<br>")))

    # Original logic
    base_chat_history.append((f"USER:{instruction}", f"AI:{synthesized_response}"))
    return base_chat_history, iteration_history
