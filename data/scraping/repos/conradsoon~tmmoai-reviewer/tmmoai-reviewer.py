import os
import openai
from datetime import datetime
import argparse
import pathlib
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def review_file(file_content, prompt, model, timeout=60):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for code reviews."},
        {"role": "system", "content": prompt},
        {"role": "user", "content": file_content}
    ]
    future = executor.submit(openai.ChatCompletion.create,
                             model=model, messages=messages)
    try:
        response = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError("Request timed out after 10 seconds")

    print(response)  # log the response
    tokens_used = response['usage']['total_tokens']
    return response.choices[0].message['content'], tokens_used


def summarize_reviews(review_contents, model):
    prompt = "Please provide a summary of the key areas for improvement and notable patterns from the given code reviews."
    concatenated_reviews = "\n".join(review_contents)
    summary, _ = review_file(concatenated_reviews, prompt, "gpt-3.5-turbo-16k")
    return summary


def main(args):
    src_directory = args.src_directory
    prompt_name = pathlib.Path(args.prompt_file).stem
    folder_name = pathlib.Path(src_directory).name
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    review_directory = f"./reviews/review_{folder_name}_{prompt_name}_{current_time}"
    os.makedirs(review_directory, exist_ok=True)

    # Read the prompt from the provided file
    with open(args.prompt_file, 'r') as f:
        prompt = f.read()

    model = args.model
    summary = {"Processed": 0, "Skipped": 0, "Files": [],
               "ReviewContents": [], "TotalTokensUsed": 0, "ReviewTexts": []}

    ignore_folders = set(args.ignore_folder)
    for root, _, files in os.walk(src_directory):
        if any(ignored_folder in root for ignored_folder in ignore_folders):
            continue
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()

                content_with_lines = "\n".join(
                    [f"{i+1}. {line}" for i, line in enumerate(content.split("\n"))])

                print(content_with_lines)
                try:
                    review_content, tokens_used = review_file(
                        content_with_lines, prompt, model)
                    summary["Processed"] += 1
                    summary["TotalTokensUsed"] += tokens_used
                    summary["ReviewContents"].append(
                        {"File": file, "Content": review_content})
                    summary["ReviewTexts"].append(review_content)

                except Exception as e:
                    print(f"Error processing {file}: {type(e).__name__} - {e}")
                    review_content = str(e)
                    summary["Skipped"] += 1
                    summary["ReviewContents"].append(
                        {"File": file, "Content": f"Error: {review_content}"})
                    summary["ReviewTexts"].append(f"Error: {review_content}")

                summary["Files"].append(
                    {"File": file, "Status": "Processed" if review_content != "file too long" else "Skipped"})

                new_directory = os.path.join(
                    review_directory, os.path.relpath(root, src_directory))
                os.makedirs(new_directory, exist_ok=True)
                with open(os.path.join(new_directory, f"{file.split('.')[0]}_review.txt"), 'w') as f:
                    f.write(review_content)

                print(
                    f"Processed {file}. Remaining: {len(files) - summary['Processed'] - summary['Skipped']}, Tokens used: {summary['TotalTokensUsed']}")

    overall_summary = summarize_reviews(summary["ReviewTexts"], model)
    summary["OverallSummary"] = overall_summary

    with open(os.path.join(review_directory, "summary.txt"), 'w') as f:
        f.write(f"Total Processed: {summary['Processed']}\n")
        f.write(f"Total Skipped: {summary['Skipped']}\n")
        f.write(f"Total Tokens Used: {summary['TotalTokensUsed']}\n")
        for file_info in summary["Files"]:
            f.write(f"{file_info['File']}: {file_info['Status']}\n")
        f.write("\nOverall Summary:\n")
        f.write(overall_summary)

    # Concatenate all the texts
    with open(os.path.join(review_directory, "concatenated_reviews.txt"), 'w') as f:
        for review_content in summary["ReviewContents"]:
            f.write(
                f"{review_content['File']}\n{review_content['Content']}\n\n")


if __name__ == "__main__":
    # print API key from environment variable
    parser = argparse.ArgumentParser(
        description="Review Java files using OpenAI's GPT model.")
    parser.add_argument("--src_directory", type=str, default=".",
                        help="Source directory containing Java files (default: current directory)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI GPT model to use for review (default: gpt-3.5-turbo-16k)")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="File containing the prompt for the model")
    parser.add_argument("--ignore-folder", type=str, nargs="*", default=[],
                        help="List of folder paths to ignore (default: none)")
    args = parser.parse_args()
    main(args)
