import os
import openai
from create_puzzle import Board
from transformers import T5Tokenizer, T5ForConditionalGeneration
import string
import random

CLUE_GEN_OPTION = "CHATGPT4"
MODEL = "gpt-3.5-turbo"

T5_MODEL_PATH = "results/word_to_clue/results_mtw/checkpoint-24000"

PROMPT = f"""Create a new york times style crossword for the following answers
Put each solution in it's own line, and do not include the answer or a substring of the answer in the clue. Use at most 2 anagrams for solutions, but try not to. There are no typos in this list, they are exactly as they should be.
Return only a numbered list with no other text. No additional text or explaination. Each solution will have one clue. Each list point will contain only the clue, with no additional text. Do not include solution word length in your clues.\n"""



def generate_chatgpt_clue(answer_list):
    answer_batch_length = 20
    answer_batches = [list(zip(*answer_list[i:i + answer_batch_length]))[-2] for i in range(0, len(answer_list), answer_batch_length)]
    #print(answer_batches)

    all_sols = []

    for batch_no, answer_batch in enumerate(answer_batches):
        curr_prompt = PROMPT + "\n".join(answer_batch) + "\n"

        #print(curr_prompt)

        openai.api_key = "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"

        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": curr_prompt}
            ]
        )

        all_sols.append(completion)
        print(completion)

        gpt_out_string = completion.choices[0].message['content']

        clues = []
        for line in gpt_out_string.split('\n'):
            parts = line.split('. ', 1)
            # Check if the line can be split into two parts and the first part is a number
            if len(parts) == 2 and parts[0].isdigit():
                clues.append(parts[1])


        for i, clue in enumerate(clues):
            answer_list[batch_no * answer_batch_length + i][-1] = clue

    return answer_list
    
def generate_t5_clue(answer_list):
    PREFIX = ""

    # load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH)

    for i, entry in enumerate(answer_list):
        word = entry[-2]

        line = PREFIX + word
        inputs = tokenizer(line, return_tensors="pt")

        output = model.generate(inputs['input_ids'], max_length=30, num_return_sequences=1)
        decoded_output = tokenizer.decode(output[0])

        answer_list[i][-1] = decoded_output

    return answer_list

def print_chatgpt_prompts(answer_list):
    answer_batch_length = 20
    answer_batches = [list(zip(*answer_list[i:i + answer_batch_length]))[-2] for i in range(0, len(answer_list), answer_batch_length)]
    #print(answer_batches)

    for batch_no, answer_batch in enumerate(answer_batches):
        curr_prompt = PROMPT + "\n".join(answer_batch) + "\n"

        print(curr_prompt)
        print()
    

def main(puzzle_path):
    with open(puzzle_path, "r") as f:
        board_string = f.read()
        board_string = board_string[:-1]

    board = Board(board_string = board_string)

    entry_list = []

    for sol_no, sol in board.sols.items():
        row = sol['row']
        col = sol['col']
        direction = sol['direction']
        answer = sol["sol"]
        entry_list.append([row, col, direction, answer, ""])

    


    if CLUE_GEN_OPTION == "T5":
        new_sols = generate_t5_clue(entry_list)
        for clue in new_sols:
            print(clue)
    elif CLUE_GEN_OPTION == "CHATGPT":
        new_sols = generate_chatgpt_clue(entry_list)
        for clue in new_sols:
            print(clue)
    elif CLUE_GEN_OPTION == "CHATGPT4":
        print_chatgpt_prompts(entry_list)
        return
    elif CLUE_GEN_OPTION == "BOTH":
        clue_t5 = generate_t5_clue(entry_list)
        clue_gpt = generate_chatgpt_clue(entry_list)
    else:
        raise ValueError("Invalid CLUE_GEN_OPTION")

    out_text = ""

    out_text += board_string + "\n"

    for sol in entry_list:
        out_text += f"{sol[0]} {sol[1]} {sol[2]} {sol[3]} {sol[4]}\n"

    # Save text under ai_puzzles/solutions
    puzzle_name = os.path.basename(puzzle_path).replace(".txt", "") + "_" + CLUE_GEN_OPTION + "_" + MODEL + "_" + ''.join(random.choice(string.ascii_uppercase) for _ in range(5)) +  ".txt"

    with open("ai_puzzles/solutions/" + puzzle_name, "w") as f:
        f.write(out_text)

if __name__ == '__main__':
    puzzle_path = "ai_puzzles/2023-08-04_08-36-04/puzzle_*LESSE***TSONE*_D2N7U.txt"

    main(puzzle_path)