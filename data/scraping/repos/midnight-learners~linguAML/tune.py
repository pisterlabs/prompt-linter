from .config import TuningSettings

def tune(tuning_settings: TuningSettings) -> None:

    # Environment
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

    # RL agent
    from linguaml.rl.agent import Agent as RLAgent
    from linguaml.tolearn.family import Family
    rl_agent = RLAgent(
        family=Family.from_name(tuning_settings.family_name),
        numeric_hp_bounds=tuning_settings.numeric_hp_bounds,
        hidden_size=tuning_settings.hidden_size,
        cont_dist_family=tuning_settings.cont_dist_family,
    )
    
    # LLM agent
    from linguaml.llm.agent import Agent as LLMAgent
    from linguaml.llm.openai.chat import OpenAIChatModel
    llm_agent = LLMAgent(
        family=Family.from_name(tuning_settings.family_name),
        numeric_hp_bounds=tuning_settings.numeric_hp_bounds,
        chat_model=OpenAIChatModel(
            model_name=tuning_settings.chat_model_name,
            temperature=tuning_settings.temperature
        )
    )

    # Advantage calulator
    from linguaml.rl.advantage import AdvantageCalculator
    advantage_calculator = AdvantageCalculator(
        moving_average_alg=tuning_settings.moving_average_alg,
        period=tuning_settings.sma_period,
        alpha=tuning_settings.ema_alpha
    )
    
    # Create a hybrid tuner
    from linguaml.tuners import HybridTuner
    from linguaml.rl.replay_buffer import ReplayBuffer
    from linguaml.tolearn.performance import PerformanceResultBuffer
    tuner = HybridTuner(
        env=env,
        rl_agent=rl_agent,
        llm_agent=llm_agent,
        replay_buffer=ReplayBuffer(
            capacity=tuning_settings.replay_buffer_capacity
        ),
        performance_result_buffer=PerformanceResultBuffer(
            capacity=tuning_settings.performance_result_buffer_capacity
        ),
        advantage_calculator=advantage_calculator,
        llm_agent_sampling_freq=tuning_settings.llm_agent_sampling_freq
    )

    # Tune!
    from torch.optim import Adam
    tuner.tune(
        n_epochs=tuning_settings.n_epochs,
        batch_size=tuning_settings.batch_size,
        min_batch_size=tuning_settings.min_batch_size,
        n_steps_for_updating_agent=tuning_settings.n_steps_for_updating_agent,
        optimizer=Adam(
            rl_agent.parameters(), 
            lr=tuning_settings.adam_lr
        ),
        ppo_epsilon=tuning_settings.ppo_epsilon
    )
