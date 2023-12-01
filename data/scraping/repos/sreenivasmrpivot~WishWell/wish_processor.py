from models import DeviceEnum, Wish
from channel.langchain.LangChainWrapper import LangChainWrapper
from channel.llamaindex.LlamaIndexWrapper import LlamaIndexWrapper
# from channel.vllm.VllmWrapper import VllmWrapper
from channel.vmwarevllmapi.VmwareVllmApiWrapper import VmwareVllmApiWrapper
import time
import argparse

def process_wish(wish: Wish):
    if wish.channel == "Langchain":
        langChainWrapper = LangChainWrapper(wish)
        grant = langChainWrapper.run()
        return grant
    elif wish.channel == "Llamaindex":
        llamaIndexWrapper = LlamaIndexWrapper(wish)
        grant = llamaIndexWrapper.run()
        return grant
    # elif wish.channel == "Vllm":
        # if wish.device == DeviceEnum.CPU:
            # raise Exception("Vllm does not work on CPU, it requires CUDA")
        # 
        # vllmWrapper = VllmWrapper(wish)
        # grant = vllmWrapper.run()
        # return grant
    elif wish.channel == "VmwareVllmApi":
        vmwareVllmApiWrapper = VmwareVllmApiWrapper(wish)
        grant = vmwareVllmApiWrapper.run()
        return grant
    else:
        raise Exception("Channel not supported")

def main(args):
    """The main function."""
    # Start time
    start_time = time.time()
    wish = Wish(device=args.device, modelLocation=args.modelLocation, documentName=args.documentName, modelName=args.modelName, channel=args.channel, vectorDatabase=args.vectorDatabase ,whisper=args.whisper)
    grant = process_wish(wish)
    print(grant)
    # End time
    end_time = time.time()

    # Total time taken
    totalTimeTaken = (end_time - start_time) * 1000
    print()
    print(f"Runtime of the program is {totalTimeTaken}")
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device", type=str, default="cpu", 
        help="Device to run the model on."
    )
    
    parser.add_argument(
        "--modelLocation", type=str, default="local", 
        help="Location of the model."
    )

    parser.add_argument(
        "--documentName", type=str, default="Business Conduct.pdf", 
        help="Name of the document."
    )

    parser.add_argument(
        "--modelName", type=str, default="Llama", 
        help="Name of the model."
    )

    parser.add_argument(
        "--channel", type=str, default="Langchain", 
        help="Name of the channel."
    )

    parser.add_argument(
        "--vectorDatabase", type=str, default="FAISS", 
        help="Name of the vector database."
    )

    parser.add_argument(
        "--whisper", type=str, default="what is Legal Holds?", 
        help="The question."
    )

    args = parser.parse_args()
    main(args)    
