import accelerate
import guidance
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

MAX_CONTEXT = 1000
OFFSET = 4
BASE_MODEL = "huggyllama/llama-7b"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"


contains_ai_program = guidance(
    """Does the following meeting transcript contain an action item? Please answer with a single word, either "Yes", "No.
Meeting transcript: {{transcript}}
Answer: {{select "answer" options=options}}"""
)
identify_ai_program = guidance(
    """Does the following meeting transcript contain an action item? Please answer with a single word, either "Yes", "No.
Meeting transcript: {{transcript}}
Answer: Yes

Now, identify the action item present in the meeting transcript.
Action Item: {{gen "text" max_tokens=20}}
Identify the assignee of that action item, this should be a persons name. If there is no clear assignee the set it to 'UNKNOWN'.
Assignee: {{gen "assignee"}}
At what time during the meeting was the action item mentioned?
Time: {{gen "time" max_tokens=7}}"""
)
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)

guidance.llm = guidance.llms.Transformers(
    model=model,
    tokenizer=tokenizer,
)


def preprocess(transcription_df):
    segments = transcription_df.apply(
        lambda x: f'{x["speaker"]} ({x["start_time"]}): {x["text"][: MAX_CONTEXT].strip()}',
        axis=1,
    ).tolist()
    chunks = [
        "\n".join(segments[i : i + OFFSET]) for i in range(0, len(segments), OFFSET)
    ]
    return chunks


def identify_ais(transcription_df, silent=True):
    chunks = preprocess(transcription_df)

    action_items = []
    for chunk in chunks:
        contains_ai = contains_ai_program(
            transcript=chunk, options=[" Yes", " No"], silent=silent
        )
        if contains_ai["answer"].strip() == "Yes":
            ai = identify_ai_program(transcript=chunk, silent=silent)
            action_items.append(
                {
                    "text": ai["text"].strip(),
                    "assignee": ai["assignee"].strip(),
                    "ts": ai["time"].strip(),
                }
            )
    return action_items
