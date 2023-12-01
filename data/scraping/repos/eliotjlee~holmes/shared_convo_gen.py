import openai
import os
import threading

openai.api_key = os.getenv("OPENAI_API_KEY")


def simulate_convo(interaction, prompt):
    """
    Simulates a conversation between two suspects using the provided prompt.

    Parameters:
        interaction (ShareInteraction): The interaction object to store the simulated conversation.
        prompt (str): The prompt to use for generating the conversation.

    Returns:
        None: Stores the simulated conversation in the interaction object.
    """
    simulated_convo = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k-0613',
        messages=[{'role': 'system', 'content': prompt}],
    )
    print(simulated_convo.choices[0].message['content'])
    interaction.set_text(simulated_convo.choices[0].message['content'])


def generate_convo(interaction, suspect_a, suspect_b, suspect_a_action, suspect_b_action, plot):
    """
    Generates a conversation between two suspects based on provided actions and context.

    Parameters:
        interaction (SharedInteraction): The interaction object to store the generated conversation.
        suspect_a (Suspect): The first suspect involved in the conversation.
        suspect_b (Suspect): The second suspect involved in the conversation.
        suspect_a_action (str): The action of suspect A at the timestamp.
        suspect_b_action (str): The action of suspect B at the timestamp.
        plot (Plot): The plot object containing details of the case.

    Returns:
        None: Calls simulate_convo function to generate and store the conversation.
    """
    prompt = (f"Timestamp: {interaction.time}. Context: A murder has taken place. Details: \n{plot.get_info}. "
              f"Suspect {suspect_a.name} details: \n\n{suspect_a.get_info()}. "
              f"Suspect {suspect_b.name} details: \n\n{suspect_b.get_info()}. "
              f"Simulate a conversation between "
              f"{suspect_a.name} and {suspect_b.name}.\n\n"
              f"This is what the two are doing at the time. Base the convo off of this:\n\n"
              f"{suspect_a_action}\n\n"
              f"{suspect_b_action}\n"
              )

    simulate_convo(interaction, prompt)


def generate_shared_interactions(plot):
    """
    Iterates over all shared interactions in the plot to generate and store conversations for consistency.

    Parameters:
        plot (object): The plot object containing details of the case and timeline.

    Returns:
        None: Generates and stores shared interactions using multi-threading.
    """
    timeline = plot.timeline
    shared_interactions_timeline = plot.shared_interactions
    name_to_action_dicts = plot.get_suspect_name_to_action()
    name_to_suspect_dict = plot.get_name_to_suspect_dict()

    threads = []

    # Iterates through, generates all shared interactions using separate threads
    for i, timestamp in enumerate(timeline):
        interaction = shared_interactions_timeline[i]
        if interaction.interaction_id is not None:
            suspect_a_name = interaction.suspect_a
            suspect_b_name = interaction.suspect_b

            suspect_a = name_to_suspect_dict[suspect_a_name]
            suspect_b = name_to_suspect_dict[suspect_b_name]

            suspect_a_action = name_to_action_dicts[i][suspect_a_name]
            suspect_b_action = name_to_action_dicts[i][suspect_b_name]

            # Start a new thread to generate the conversation
            t = threading.Thread(target=generate_convo,
                                 args=(interaction, suspect_a, suspect_b, suspect_a_action, suspect_b_action, plot))
            threads.append(t)
            t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()
