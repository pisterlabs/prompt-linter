# Imports from this package
from .config import TuningSettings

def tune(tuning_settings: TuningSettings) -> None:
    
    # Set up the environment
    from linguaml.rl.env import Env
    from linguaml.data.dataset import load_dataset
    env = Env(
        datasets=[
            load_dataset(name=name)
            for name in tuning_settings.dataset_names
        ],
        performance_metric=tuning_settings.performance_metric,
        lookback=tuning_settings.lookback,
        fitting_time_limit=tuning_settings.fitting_time_limit,
        random_state=tuning_settings.random_state
    )
    
    # LLM Agent
    from linguaml.llm.agent import Agent
    from linguaml.tolearn.family import Family
    from linguaml.llm.openai.chat import OpenAIChatModel
    agent = Agent(
        family=Family.from_name(tuning_settings.family_name),
        numeric_hp_bounds=tuning_settings.numeric_hp_bounds,
        chat_model=OpenAIChatModel(
            model_name=tuning_settings.chat_model_name,
            temperature=tuning_settings.temperature
        )
    )
    
    # Create an LLM tuner
    from linguaml.tuners import LLMTuner
    from linguaml.rl.replay_buffer import ReplayBuffer
    from linguaml.tolearn.performance import PerformanceResultBuffer
    tuner = LLMTuner(
        env=env,
        agent=agent,
        replay_buffer=ReplayBuffer(
            capacity=tuning_settings.replay_buffer_capacity
        ),
        performance_result_buffer=PerformanceResultBuffer(
            capacity=tuning_settings.performance_result_buffer_capacity
        )
    )
    
    # Tune!
    tuner.tune(
        n_epochs=tuning_settings.n_epochs
    )
