import torch
import argparse
import time
from fastchat.model import load_model
from fastchat.conversation import get_conv_template
from PromptGenerator import PromptGenerator
import os

# python vicuna_cli.py --knowledge-base
# python vicuna_cli.py --knowledge-base --load-8bit

# Parameters for prompt generation
k = 5
chunk_length = 128


# Parameters for saving the log file when inference is performed
para = 7
DATA_PATH = "data/staff_a"
MODELNAME = "vicuna-13b-v1.3" if para == 13 else "vicuna-7b-v1.5"
EMBEDDING = "InstructorEmbeddings"
TRUNCATION = True
DOCUMENT = DATA_PATH.split("/")[-1]
TIME = time.strftime("%m-%d-%H-%M-%S", time.localtime())

def get_log_dir(modelname, embedding, truncation, document, k=k, chunk_length=chunk_length, time = TIME):
    trunc = "Trunc" if truncation else "noTrunc"
    dir = f"{time}_{modelname}_{embedding}_{trunc}_k={k}_cl={chunk_length}_{document}"
    return dir


def main(args): 
    model, tokenizer = load_model(
        args.vicuna_dir,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        debug=args.debug,
    )
    
    # Record the conversation in a log file
    log_text = "```json\n"

    conv = get_conv_template("vicuna_v1.1")
    if args.knowledge_base:
        prompt_generator = PromptGenerator(knowledge_dir = DATA_PATH, k = k, chunk_length = chunk_length)

    while True:
        question = input("\nUSER: ")
        
        if question == "!!reset":
            # Reset the conversation without any history context
            
            conv = get_conv_template("vicuna_v1.1")
            print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    RESET    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            log_text += "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    RESET    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n"
            continue
            
        elif question == "!!exit":
            # Exit the conversation
            
            print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    EXIT    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            log_text += "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    EXIT    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n"
            break
        
        log_text += f"\nUSER: {question}\n"
        
        if args.knowledge_base:
            msg, log_text = prompt_generator.get_prompt([question], log_text)
            # print(msg)
        else:
            msg = question

        # Only append the user message to the history context
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Uncomment the following lines to print the history context
        for i in range(len(conv.messages)):
            print(f"{conv.messages[i]}")
            log_text += f"{conv.messages[i]}\n"
        print("==============================================")
        log_text += "==============================================\n"

        input_ids = tokenizer([prompt]).input_ids

        output_ids = model.generate(
            torch.as_tensor(input_ids).to(args.device),
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

        conv.correct_message(question)
        print(f"\nASSISTANT: {outputs}")
        print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        
        log_text += f"\nASSISTANT: {outputs}\n"
        log_text += "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"    
        log_text += ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n"
        
    # Save the log file, if the directory does not exist, create it
    log_text += '\n```'
    log_dir = get_log_dir(MODELNAME, EMBEDDING, TRUNCATION, DOCUMENT, k, chunk_length, TIME)
    if not os.path.exists("inference_examples"):
        os.mkdir("inference_examples")
    with open(f"inference_examples/{log_dir}.txt", "w", encoding="utf-8") as f:
        f.write(log_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vicuna-dir",
        type=str,
        default="../"+MODELNAME,
        help="The path to the weights",
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda"
    )
    parser.add_argument("--knowledge-base", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
