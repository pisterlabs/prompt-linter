# Libraries
import argparse
import cohere
import os
import pytesseract
import imutils
import cv2
from difflib import SequenceMatcher
from pathlib import Path
from typing import Union
from dotenv import load_dotenv
from PIL import Image
import numpy as np


# Main class
class Gen:
    """
    Main inference class
    """
    def __init__(self): 
        # Variables
        self.model='small'
        load_dotenv()
        self.COHERE_APIKEY = os.getenv('COHERE_APIKEY')
        self.co = cohere.Client(self.COHERE_APIKEY)

    def is_good_word(self, s):
        if len(s) == 0:
            return False
        if len(s) == 1 and s.lower() not in ['a', 'i']:
            return False
        return True
        
    def predict(self, image: Union[str, Path, Image.Image], max_tokens: int) -> str:
        if isinstance(image, Image.Image): img = np.asarray(image)
        else: img = cv2.imread(image)
        img = imutils.resize(img, width=500, height=500)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
        results = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='eng')

        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]

            text = results["text"][i]
            conf = int(results["conf"][i])

            if conf > 0:
                text = "".join(text).strip()
                cv2.rectangle(img,
                            (x, y),
                            (x + w, y + h),
                            (0, 0, 255), 2)

        prompt = ' '.join([i for i in results['text'] if self.is_good_word(i)])

        response = self.co.generate(prompt=prompt, max_tokens=max_tokens, model=self.model)
        return prompt, response.generations[0].text

    def similar(self, a, b):
        # return between 0 and 1
        # 1 is identical, 0 is completely different
        return SequenceMatcher(None, a, b).ratio()

    def actual_text(self, path):
        lines = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line[3:]      # remove "1: "
                line = line.strip()
                lines.append(line)

        return ' '.join(lines)

    # i: 1-21
    def comparing(self, i):
        label = self.actual_text(f'./training-strips/labels/cartoon{i}.txt')
        ocr = self.predict(f'./training-strips/images/cartoon{i}.png')
        return self.similar(label, ocr)


# Running model
def main():
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--image", type=str, required=True)

    parser.add_argument("--max_tokens", type=int, required=True)

    args = parser.parse_args()

    # Answering question
    pipeline = Gen()
    prompt, gen = pipeline.predict(args.image, args.max_tokens)

    print("OCR:" + prompt + '\n' + "Generated text:" + gen)
    
    """
    a = self.actual_text("./training-strips/labels/cartoon1.txt")
    print(a)

    accs = []
    for i in range(1, 22):
        acc = pipeline.comparing(i)
        print(f"cartoon{i}: {acc}")
        accs.append(acc)

    print(f"\naverage: {sum(accs) / len(accs)}")
    """


if __name__ == '__main__':
    main()