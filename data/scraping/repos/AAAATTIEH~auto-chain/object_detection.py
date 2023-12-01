from langchain.tools import BaseTool
from transformers import  DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Image Path Finder Tool must be used before using this tool" \
                   "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return "Image path must be found. Use the ImagePathFinder tool"
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

