import os
import pickle
import threading
import time
from queue import Queue
from multiprocessing import Manager, Process
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
import argparse

# Define a template for translation.
TRANSLATION_TEMPLATE = """Translate the following text to the {language} language. Do not print any extra text and only give me the translated text as best as possible.\n\n"{content}".\n\nTranslation:"""

task_queue = Queue()  # Queue for tasks

def writer(output_file, queue):
    query_inputs = {}
    llm_responses = {}
    llm_full_responses = {}

    # Load existing data from the file
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            previous_data = pickle.load(f)
            query_inputs = previous_data["query_inputs"]
            llm_responses = previous_data["llm_responses"]
            llm_full_responses = previous_data["llm_full_responses"]

    while True:
        item = queue.get()
        if item is None:  # Check for the sentinel value
            break

        (
            local_query_inputs,
            local_llm_responses,
            local_llm_full_responses,
        ) = item

        # Update existing data with new data
        query_inputs.update(local_query_inputs)
        llm_responses.update(local_llm_responses)
        llm_full_responses.update(local_llm_full_responses)

        # Save the combined data
        with open(output_file, "wb") as f:
            store_df = {
                "query_inputs": query_inputs,
                "llm_responses": llm_responses,
                "llm_full_responses": llm_full_responses,
            }
            pickle.dump(store_df, f)

def chat_query(prompt_content, target_language="English"):
    # Modify the prompt to use the translation template
    prompt = TRANSLATION_TEMPLATE.format(
        content=prompt_content, language=target_language
    )
    
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.completions.create(
        model="claude-2.0",
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
        max_tokens_to_sample=16384,
    )
    answer = response.completion
    return answer, response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def process_sample(sample_id, data_frame, field, target_language, queue, pbar):
    local_query_inputs = {}
    local_llm_responses = {}
    local_llm_full_responses = {}

    try:
        prompt_content = data_frame.loc[sample_id][field]
        x, y = chat_query(prompt_content, target_language)

        local_query_inputs[sample_id] = prompt_content
        local_llm_responses[sample_id] = x
        local_llm_full_responses[sample_id] = y

        # Pass a tuple of dictionaries to the queue
        queue.put((local_query_inputs, local_llm_responses, local_llm_full_responses))
    except Exception as e:
        print("error for sample_id: ", sample_id)
        print(e)
        raise
    finally:
        pbar.update(1)  # Update the progress bar

def pickle_to_csv(pickle_file, csv_output_file):
    # Load the pickle file
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    # Convert the loaded data into a DataFrame
    df = pd.DataFrame(
        {
            "query_inputs": pd.Series(data["query_inputs"]),
            "responses": pd.Series(data["llm_responses"]),
            "full_responses": pd.Series(
                {k: str(v) for k, v in data["llm_full_responses"].items()}
            ),
        }
    )

    df.to_csv(csv_output_file, index=False)

def worker():
    while True:
        task = task_queue.get()
        if task is None:  # Check for the sentinel value to terminate
            break

        sample_id, data_frame, field, target_language, queue, pbar = task
        process_sample(sample_id, data_frame, field, target_language, queue, pbar)

def main():
    parser = argparse.ArgumentParser(description="Process a CSV file with Claude's GPT.")
    parser.add_argument("csv_file_path", type=str, help="Path to the CSV file.")
    parser.add_argument("field", type=str, help="Field in the CSV to process with GPT.")
    parser.add_argument("--sample_size", type=int, default=-1, help="Number of samples to process. Default is -1 (all samples).")
    parser.add_argument("--output_file", type=str, default="llm_responses.pkl", help='Output file name. Defaults to "llm_responses.pkl".')
    parser.add_argument("--target_language", type=str, default="English", help='Target language for the translation. Defaults to "English".')
    parser.add_argument("--threadpool_size", type=int, default=10, help="Number of threads in the ThreadPool. Defaults to 10.")
    args = parser.parse_args()

    data_frame = pd.read_csv(args.csv_file_path)
    if os.path.exists(args.output_file):
        with open(args.output_file, "rb") as f:
            previous_data = pickle.load(f)
            prev_query_inputs = previous_data["query_inputs"]
    else:
        prev_query_inputs = {}

    requesed_ids = list(prev_query_inputs.keys())
    remaining_sets = data_frame[~data_frame.index.isin(requesed_ids)]
    sample_bag = remaining_sets.index.tolist() if args.sample_size == -1 else remaining_sets.sample(args.sample_size).index.tolist()

    with Manager() as manager:
        queue = manager.Queue()  # Create a shared queue
        pbar = tqdm(total=len(sample_bag), dynamic_ncols=True)

        writer_process = Process(target=writer, args=(args.output_file, queue))
        writer_process.start()

        # Setup workers
        threads = []
        for _ in range(args.threadpool_size):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Dispatch tasks
        for sample_id in sample_bag:
            task_queue.put((
                sample_id,
                data_frame,
                args.field,
                args.target_language,
                queue,
                pbar,
            ))
            time.sleep(1)

        # Terminate workers
        for _ in range(args.threadpool_size):
            task_queue.put(None)
        for t in threads:
            t.join()

        queue.put(None)  # Indicate that all data has been processed
        writer_process.join()  # Wait for the writer process to finish

    pbar.close()
    pickle_to_csv(args.output_file, args.output_file.replace(".pkl", ".csv"))

if __name__ == "__main__":
    main()
