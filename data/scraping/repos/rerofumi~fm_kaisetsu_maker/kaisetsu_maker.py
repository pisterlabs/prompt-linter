import os
import argparse
from src.openai_api_bridge import OpenAIAPIBridge
from src.explanation import Explanation

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    args = parse_args()
    #
    api = OpenAIAPIBridge(api_key)
    generator = Explanation(api, "./resource", "./output", args.use_voicevox)
    print(f"Question: {args.question}")
    print(f"Image Only: {args.image_only}")
    print(f"Keep Image: {args.keep_image}")
    generator.generate(args.question, args.image_only, args.keep_image)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Your question.")
    parser.add_argument("--image-only", "-i", action="store_true", required=False, help="Perform only image output, do not generate video.")
    parser.add_argument("--keep-image", "-k", action="store_true", required=False, help="Do not generate images.")
    parser.add_argument("--use-voicevox", "-v", action="store_true", required=False, help="Use VOICEVOX engine.")
    return parser.parse_args()

if __name__ == '__main__':
    main()
