import os
import openai
import json
import time
import argparse
import json
from datetime import datetime
from io import BytesIO
from langchain.schema.chat import ChatSession
from langchain.schema.messages import HumanMessage


# Note: This fails due to 'content' not existing for some messages
#
# from langchain.chat_loaders.facebook_messenger import (
#     SingleFileFacebookMessengerChatLoader,
# )
from langchain.chat_loaders.utils import merge_chat_runs, map_ai_messages
from langchain.adapters.openai import convert_messages_for_finetuning


def main():
    # Check that OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY env var must be set: export OPENAI_API_KEY=YOUR_KEY_HERE")

    parser = argparse.ArgumentParser(description="Fine tunes a model for both participants from messenger")
    parser.add_argument("msg_dir", type=str, help="Directory containing messenger jsons")
    parser.add_argument("--only", type=str, help="Only make a model for this participant")

    args = parser.parse_args()

    # Combine all the json files into one
    raw_messages = []
    participants = None
    for json_file in os.listdir(args.msg_dir):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join(args.msg_dir, json_file)
        print("Processing ", json_path)
        with open(json_path, "r") as f:
            chat_info = json.load(f)
            if participants is None:
                participants = chat_info["participants"]
            else:
                if participants != chat_info["participants"]:
                    raise ValueError("Participants don't match")

            for msg in chat_info["messages"]:
                # Ignore calls
                if "call_duration" in msg:
                    continue

                # Check if content, sender_name, and timestamp_ms exist
                if not all(k in msg for k in ("content", "sender_name", "timestamp_ms")):
                    continue
                raw_messages.append(msg)

    sorted_data = sorted(raw_messages, key=lambda x: x["timestamp_ms"])

    # for msg in sorted_data[:36]:
    #     # convert timestamp_ms to datetime
    #     dt = datetime.fromtimestamp(msg["timestamp_ms"] / 1000)
    #     print(msg["sender_name"], ":\t", dt, msg["content"])
    # return

    # Split the messages by if there is any gap longer than 6 hours
    all_sessions = []
    current_messages = []
    prev_msg_time = sorted_data[0]["timestamp_ms"]
    num_too_long = 0
    for raw_msg in sorted_data:
        if len(raw_msg["content"]) > 1000:
            num_too_long += 1
            continue

        if raw_msg["timestamp_ms"] - prev_msg_time > 1000 * 60 * 60 * 6:
            all_sessions.append(ChatSession(messages=current_messages))
            current_messages = []

        current_messages.append(
            HumanMessage(content=raw_msg["content"], additional_kwargs={"sender": raw_msg["sender_name"]})
        )
        prev_msg_time = raw_msg["timestamp_ms"]

    merged_sessions = list(merge_chat_runs(all_sessions))
    filtered_sessions = []
    for session in merged_sessions:
        if len(session["messages"]) > 4:
            filtered_sessions.append(session)

    print("Messages that were too long", num_too_long)
    print(len(merged_sessions), len(filtered_sessions))

    if args.only:
        print("Limiting participants to --only filter")
        participants = list(filter(lambda x: x["name"] == args.only, participants))

    for participant in participants:
        participant_name = participant["name"]
        print(participant)
        alternating_sessions = map_ai_messages(filtered_sessions, participant_name)

        training_data = convert_messages_for_finetuning(alternating_sessions)
        print(f"Prepared {len(training_data)} dialogues for training")
        print(training_data[:10])

        my_file = BytesIO()
        for m in training_data:
            my_file.write((json.dumps({"messages": m}) + "\n").encode("utf-8"))

        my_file.seek(0)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        training_file = openai.File.create(file=my_file, purpose="fine-tune")

        # OpenAI audits each training file for compliance reasons.
        # This make take a few minutes
        status = openai.File.retrieve(training_file.id).status
        start_time = time.time()
        while status != "processed":
            print(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
            time.sleep(5)
            status = openai.File.retrieve(training_file.id).status
        print(f"File {training_file.id} ready after {time.time() - start_time:.2f} seconds.")

        job = openai.FineTuningJob.create(
            training_file=training_file.id,
            model="gpt-3.5-turbo",
        )

        status = openai.FineTuningJob.retrieve(job.id).status
        start_time = time.time()
        while status != "succeeded":
            print(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
            time.sleep(5)
            job = openai.FineTuningJob.retrieve(job.id)
            status = job.status

        print("Use this model id to talk to your model:")
        print(job.fine_tuned_model)


if __name__ == "__main__":
    main()
