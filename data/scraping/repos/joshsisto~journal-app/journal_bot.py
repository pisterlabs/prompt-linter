import openai
import os
from datetime import datetime,timedelta
import glob

from utilities import get_today, get_now, red, blue, bold 

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_prompt_and_conversation():
    # Load prompt
    with open('./prompts/assistant_prompt.txt', 'r') as f:
        assistant_prompt = f.read()

    # Load .all file
    all_file_path = f"./logs/{get_today()}/{get_today()}.all"
    with open(all_file_path, 'r') as f:
        all_conversation = f.read()

    # Concatenate the prompt and the conversation
    conversation = assistant_prompt + "\n" + all_conversation
    return conversation

def chatbot():
    # Get the combined prompt and conversation
    conversation = get_prompt_and_conversation()

    # Create an initial system message with the conversation
    messages = [
        {"role": "system", "content": conversation},
    ]

    timestamp_start = datetime.now()
    timestamp_str = timestamp_start.strftime("%Y-%m-%d_%H-%M-%S")

    filename = f'./logs/{get_today()}/{timestamp_str}.chat'

    with open(filename, 'w') as f:
        f.write(f"Conversation started at: {timestamp_str}\n\n")

        # Send the messages to the assistant and get the response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6
        )
        assistant_message = response.choices[0].message['content']
        messages.append({"role": "assistant", "content": assistant_message})
        f.write("Assistant: " + assistant_message + "\n\n")
        print("Assistant: ", blue(assistant_message))

        while True:
            user_message = input(bold(red("You: ")))

            if user_message.lower() == "quit":
                timestamp_end = datetime.now()
                f.write(f"\nConversation ended at: {timestamp_end.strftime('%Y-%m-%d_%H-%M-%S')}")
                duration = timestamp_end - timestamp_start
                f.write(f"\nDuration of conversation: {str(duration)}\n")
                break

            messages.append({"role": "user", "content": user_message})
            f.write("You: " + user_message + "\n\n")

            # Send the messages to the assistant and get the response
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.8,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6
            )
            assistant_message = response.choices[0].message['content']
            messages.append({"role": "assistant", "content": assistant_message})
            f.write("Assistant: " + assistant_message + "\n\n")
            print("Assistant: ", blue(assistant_message))


def summarize_all_files():
    # Load the summarization prompt
    with open('./prompts/summary_prompt.txt', 'r') as f:
        summary_prompt = f.read()

    # Scan the logs directory for all .all files
    all_files = glob.glob("./logs/*/*.all")

    for file_path in all_files:
        # Check if corresponding .summ file already exists
        summary_filename = file_path.replace('.all', '.summ')
        if os.path.exists(summary_filename):
            print(f"Skipping {file_path} as {summary_filename} already exists.")
            continue

        # Read the content of the .all file
        with open(file_path, 'r') as f:
            content = f.read()

        # Concatenate the prompt and the content
        conversation = summary_prompt + "\n" + content

        # Prepare the initial message to send to the assistant
        messages = [
            {"role": "system", "content": conversation},
        ]

        # Get the summary from the assistant
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6
        )
        summary = response.choices[0].message['content']

        # Save the summary to a .summ file
        with open(summary_filename, 'w') as f:
            f.write(summary)

    print("Summarization completed for all applicable .all files.")


def weekly_summary():
    # Load the weekly summarization prompt
    with open('./prompts/weekly_summary_prompt.txt', 'r') as f:
        weekly_summary_prompt = f.read()

    # Get all .summ files and sort them
    summ_files = sorted(glob.glob("./logs/*/*.summ"))

    while summ_files:
        # Take last 7 .summ files for a week
        weekly_files = summ_files[-7:]
        del summ_files[-7:]

        # Aggregate content from these .summ files
        aggregated_content = ""
        for file_path in weekly_files:
            with open(file_path, 'r') as f:
                aggregated_content += f.read() + "\n\n"

        # Concatenate the prompt and the aggregated content
        conversation = weekly_summary_prompt + "\n" + aggregated_content

        # Prepare the initial message to send to the assistant
        messages = [
            {"role": "system", "content": conversation},
        ]

        # Get the weekly summary from the assistant
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6
        )
        summary = response.choices[0].message['content']

        # Extract the date range based on the filenames in the chunk
        start_date_str = os.path.basename(weekly_files[0]).replace('.summ', '')
        end_date_str = os.path.basename(weekly_files[-1]).replace('.summ', '')
        date_range_str = f"{start_date_str}_to_{end_date_str}"

        # Save the weekly summary to a .week file in the root of the /logs/ folder
        weekly_filename = f"./logs/{date_range_str}.week"
        with open(weekly_filename, 'w') as f:
            f.write(summary)

        print(f"Weekly summary saved to {weekly_filename}.")


def count_tokens(text):
    """Utility function to count tokens in a given text."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "user", "content": text}],
    )
    return response.usage['total_tokens']


def total_summary():
    # Load the total summarization prompt
    with open('./prompts/total_summary_prompt.txt', 'r') as f:
        total_summary_prompt = f.read()

    prompt_tokens = count_tokens(total_summary_prompt)
    max_tokens = 10000 - prompt_tokens  # Adjusting for the prompt's token count

    # Get all .summ files and sort them
    summ_files = sorted(glob.glob("./logs/*/*.summ"))

    aggregated_content = ""
    token_count = 0
    start_file = summ_files[0]

    for file_path in summ_files:
        with open(file_path, 'r') as f:
            content = f.read()
            tokens = count_tokens(content)
            
            if token_count + tokens > max_tokens:
                # Save the aggregated content as .sum and reset
                end_file = file_path
                save_summary(aggregated_content, start_file, end_file, total_summary_prompt)
                
                # Reset aggregation and token count
                aggregated_content = content + "\n\n"
                token_count = tokens
                start_file = file_path
            else:
                aggregated_content += content + "\n\n"
                token_count += tokens

    # Handle any remaining content after the loop ends
    if aggregated_content:
        end_file = summ_files[-1]
        save_summary(aggregated_content, start_file, end_file, total_summary_prompt)


def save_summary(aggregated_content, start_file, end_file, total_summary_prompt):
    """Utility function to save the summary based on content and date range."""
    conversation = total_summary_prompt + "\n" + aggregated_content
    messages = [{"role": "system", "content": conversation}]
    
    # Adjusting the max tokens to ensure total tokens (input + output) is within model's limit
    max_response_tokens = 14385 - len(conversation.split())  # Assuming one word = one token for simplicity
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0.8,
        max_tokens=max_response_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6
    )
    summary = response.choices[0].message['content']

    start_date_str = os.path.basename(start_file).replace('.summ', '')
    end_date_str = os.path.basename(end_file).replace('.summ', '')
    date_range_str = f"{start_date_str}_to_{end_date_str}"

    total_filename = f"./logs/{date_range_str}.sum"
    with open(total_filename, 'w') as f:
        f.write(summary)

    print(f"Total summary saved to {total_filename}.")
