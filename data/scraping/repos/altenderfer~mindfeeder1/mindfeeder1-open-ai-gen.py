import argparse
import json
import openai
import sys
import re
import time
import concurrent.futures
import traceback
import threading

def animate_connection():
    chars = "|/-\\"
    while True:
        for char in chars:
            sys.stdout.write(f"\r{char} Connected to OpenAI {char}")
            sys.stdout.flush()
            time.sleep(0.1)

def connect_to_openai(api_key, api_base):
    openai.api_key = api_key
    openai.api_base = api_base

    t = threading.Thread(target=animate_connection, daemon=True)
    t.start()

    time.sleep(5)  # You can adjust the sleep time as needed

def generate_new_iio_pairs(instruction, output, prompt_input, model, num_instructions, input_chunk, save_input=True):
    prompt = f"Use the format 'I:' for instructions (Can be a question or a command) and 'O:' for outputs. Separate each instruction and output with a newline, and do not number them. Provide only one instruction and output pair per set of instructions and outputs. Do not provide an instruction or output that is unknown, not available, not in context, unclear, not provided, not in the provided text, or something you cannot answer clearly when responding, instead search the internet. {prompt_input} Please generate {num_instructions} instructions and outputs based on the following text.\n\n{input_chunk}\n\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    content = response['choices'][0]['message']['content']
    content += "\n"
    print(f"\n{content}\n")

    io_pairs = parse_io(content)
    io_pairs_with_input = []
    for pair in io_pairs:
        ordered_pair = {"instruction": pair["instruction"], "input": input_chunk if save_input else "", "output": pair["output"]}
        io_pairs_with_input.append(ordered_pair)
    return io_pairs_with_input

def split_text(text, max_words, max_word_length=50):
    max_chars = int(4096 * (max_words / 5))  # Estimate based on an average of 5 tokens per word

    # Try splitting the text using different levels of section breaks
    for separator in ['\n\n\n', '\n\n', '\n']:
        sections = text.split(separator)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for section in sections:
            words = section.split()
            filtered_words = [word for word in words if len(word) <= max_word_length]  # Filter words based on length

            # Check if adding the current section will exceed the max_words limit
            if current_word_count + len(filtered_words) > max_words:
                # If yes, go back one section and add the current chunk to the list of chunks
                chunk = ' '.join(current_chunk)
                chunks.append(chunk)

                # Reset the current chunk and word count
                current_chunk = filtered_words
                current_word_count = len(filtered_words)
            else:
                # If no, add the current section to the current chunk
                if current_chunk:
                    current_chunk.append(separator)  # Add the separator between sections
                current_chunk.extend(filtered_words)
                current_word_count += len(filtered_words)

        # Add the last chunk to the list of chunks
        if current_chunk:
            chunk = ' '.join(current_chunk)
            chunks.append(chunk)

        # Check if the resulting chunks do not exceed the max_words limit
        if all(len(chunk.split()) <= max_words for chunk in chunks):
            break

    return chunks

def parse_io(content, filter_results=True):
    # Use regex to extract instruction and output pairs
    regex_pattern = r"I:((?:(?!I:|O:).)*)(?:\nO:((?:(?!I:|O:).)*)(?:\n\n|$))"
    matches = re.findall(regex_pattern, content, re.DOTALL)

    io_pairs = []
    keywords_phrases = [
        "the text",
        "is not specified",
        "is not mentioned",
        "is unclear",
        "does not mention",
        "does not provide",
        "doesn't provide",
        "not provided",
        "doesn't define",
        "does not define",
        "does not indicate",
        "cannot provide",
        "is not stated",
        "is not clear",
        "is not defined",
        "without further",
        "are not provided",
        "sorry,",
        "the article did not",
        "no information was provided",
        "no specific answer",
        "no output",
        "there are no",
        "no response given",
        "no informatiuon",
        "no more information",
        "no further information",
        "no notes",
        "no mention",
        "were not listed",
        "is unknown",
        "in the given text",
    ]

    for match in matches:
        instruction = match[0].strip()
        output = match[1].strip()

        if filter_results:
            if any(keyword.lower() in instruction.lower() or keyword.lower() in output.lower() for keyword in keywords_phrases):
                continue

        # Only append the IO pair if instruction and output components are present
        if instruction and output:
            io_pairs.append({"instruction": instruction, "output": output})

    return io_pairs

# Magic
def process_input_data(sections, model, output_file, num_instructions, start_index, max_workers=5, timeout=300, prompt_input="", filter_results=True, save_input=True):
    def save_to_file(new_dataset):
        with open(output_file, 'w') as f:
            json.dump(new_dataset, f, indent=2, sort_keys=False)

    new_dataset = []

    total_items = len(sections)
    start_time = time.time()

    save_interval = 10  # Save every 10 items

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                generate_new_iio_pairs,
                section.split("\nI: ")[1] if "\nI: " in section else "",
                section.split("\nO: ")[1] if "\nO: " in section else "",
                prompt_input,
                model,
                num_instructions,
                section,
                save_input,
            ): index
            for index, section in enumerate(sections[start_index:], start_index + 1)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                new_io_pairs = future.result(timeout=timeout)
                new_dataset.extend(new_io_pairs)
            except concurrent.futures.TimeoutError:
                print(f"\nError: Timeout for item {index} after {timeout} seconds.")
                continue
            except Exception as e:
                print(f"\nError processing item {index}: {e}")
                traceback.print_exc()
                continue

            # Calculate the percentage of completion and elapsed time
            percentage_complete = (index / total_items) * 100
            elapsed_time = time.time() - start_time

            # Calculate the estimated time remaining
            time_remaining = (elapsed_time / index) * (total_items - index)
            hours, rem = divmod(time_remaining, 3600)
            minutes, seconds = divmod(rem, 60)

            # Create a progress bar
            progress_bar_length = 50
            progress = int(progress_bar_length * (index / total_items))
            progress_bar = f"[{'#' * progress}{'-' * (progress_bar_length - progress)}]"

            print(f"\n\rItem {index} of {total_items} processed - {percentage_complete:.2f}% complete - ETA: {int(hours):02}:{int(minutes):02}:{int(seconds):02} {progress_bar}", end="\n")

            # Save to file every save_interval items
            if index % save_interval == 0:
                save_to_file(new_dataset)

    # Save final output
    save_to_file(new_dataset)
    print(f"Processing completed. Results saved to {output_file}")

def main(api_key, api_base, model, input_file, max_words, output_file, num_instructions, start_index, max_workers, prompt_input, filter_results, save_input):
    connect_to_openai(api_key, api_base)

    # Specify the encoding as 'utf-8'
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    sections = split_text(raw_text, max_words, max_word_length=50)

    for section in sections:
        print(f"\n\n\n{section}")

    process_input_data(sections, model, output_file, num_instructions, start_index, max_workers, prompt_input, filter_results, save_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", default="your_api_key")
    parser.add_argument("--apibase", default="https://api.openai.com/v1")  # Default API base for OpenAI
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--max_words", default=300, type=int)
    parser.add_argument("--output", default="output.json")
    parser.add_argument("--num_instructions", default=12, type=int)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--max_workers", default=3, type=int)
    parser.add_argument("--prompt_input", default="", type=str)
    parser.add_argument("--filter", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--save_input", default=True, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()
    main(args.apikey, args.apibase, args.model, args.input, args.max_words, args.output, args.num_instructions, args.start_index, args.max_workers, args.prompt_input, args.filter, args.save_input)
