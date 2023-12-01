import cv2
import os
import torch
import openai
import functools
import numpy as np
import face_detection
import io, tokenize
from torchvision import transforms
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter

from .nms import nms
from .api import API


def parse_step(step_str,partial=False): # ANSWER1=EVAL(image=IMAGE,expr=f"'top' if {ANSWER0} > 0 else 'bottom'",object='vehicle')
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    # print(tokens)
    output_var = tokens[0].string # ANSWER1
    step_name = tokens[2].string # EVAL
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']] # image IMAGE ...
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string # dict: image -> IMAGE
    parsed_result['args'] = args
    return parsed_result


class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def execute(self,expr):
        if 'xor' in expr:
            expr = expr.replace('xor','!=')

        step_output = eval(expr)

        print("EVAL")
        print(step_output)
        return step_output


class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def execute(self,output):
        return output


class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def predict(self,img,question):
        return API.vqa(img,question)
        # return API.vqa_short(img,question)

    def execute(self,img,question):
        answer = self.predict(img,question)
        return answer


class LocInterpreter():
    """
    Input:
        img: an image object
        obj_name: an object string
    Output:
        selected_boxes: a list of bounding boxes
    """

    step_name = 'LOC'

    def __init__(self,thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.thresh = thresh
        self.nms_thresh = nms_thresh

    def predict(self,img,obj_name):
        return API.loc(img,obj_name,self.thresh,self.nms_thresh)
        # return API.find(img,obj_name,glip_thresh=0.6)

    def top_box(self,img):
        w,h = img.size
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    def execute(self,img,obj_name):
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        else:
            bboxes = self.predict(img,obj_name)
        return bboxes


class Loc2Interpreter(LocInterpreter):

    def execute(self,img,obj_name):
        bboxes = self.predict(img,obj_name)

        objs = []
        for box in bboxes:
            objs.append(dict(
                box=box,
                category=obj_name
            ))

        return objs


class CountInterpreter():
    """
	Input:
        box: a list of bounding boxes
    Output:
        number: number of objects
	Examples:
	ANSWER0=COUNT(box=BOX1)
	"""
    step_name = 'COUNT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def execute(self,boxes):
        count = len(boxes)
        return count


class CropInterpreter():
    """
	crop a patch  of the image identified by the bounding box
	Input:
        image: an image
        box: a box
    Output:
        image: an cropped image
	Examples:
	IMAGE0=CROP(image=IMAGE,box=BOX0)
	"""
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def execute(self,img,boxes):
        if len(boxes) > 0:
            box = boxes[0]
            box = self.expand_box(box, img.size)
            out_img = img.crop(box)
        else:
            box = []
            out_img = img

        return out_img


class CropRightOfInterpreter(CropInterpreter):
    step_name = 'CROP_RIGHTOF'

    def right_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [cx,0,w-1,h-1]

    def execute(self,img,boxes):
        if len(boxes) > 0:
            box = boxes[0]
            right_box = self.right_of(box, img.size)
        else:
            w,h = img.size
            box = []
            right_box = [int(w/2),0,w-1,h-1]
        out_img = img.crop(right_box)

        return out_img


class CropLeftOfInterpreter(CropInterpreter):
    step_name = 'CROP_LEFTOF'

    def left_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [0,0,cx,h-1]

    def execute(self,img,boxes):
        if len(boxes) > 0:
            box = boxes[0]
            left_box = self.left_of(box, img.size)
        else:
            w,h = img.size
            box = []
            left_box = [0,0,int(w/2),h-1]
        out_img = img.crop(left_box)

        return out_img


class CropAboveInterpreter(CropInterpreter):
    step_name = 'CROP_ABOVE'

    def above(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,0,w-1,cy]

    def execute(self,img,boxes):
        if len(boxes) > 0:
            box = boxes[0]
            above_box = self.above(box, img.size)
        else:
            w,h = img.size
            box = []
            above_box = [0,0,int(w/2),h-1]
        out_img = img.crop(above_box)

        return out_img

class CropBelowInterpreter(CropInterpreter):
    step_name = 'CROP_BELOW'

    def below(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,cy,w-1,h-1]

    def execute(self,img,boxes):
        if len(boxes) > 0:
            box = boxes[0]
            below_box = self.below(box, img.size)
        else:
            w,h = img.size
            box = []
            below_box = [0,0,int(w/2),h-1]
        out_img = img.crop(below_box)

        return out_img

class CropFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_FRONTOF'

class CropInFrontInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONT'

class CropInFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONTOF'

class CropBehindInterpreter(CropInterpreter):
    step_name = 'CROP_BEHIND'


class CropAheadInterpreter(CropInterpreter):
    step_name = 'CROP_AHEAD'


class SegmentInterpreter():
    step_name = 'SEG'

    def execute(self,img):
        objs = API.segment(img)

        return objs


class SelectInterpreter():
    step_name = 'SELECT'

    def query_string_match(self,objs,q):
        obj_cats = [obj['category'] for obj in objs]
        q = q.lower()
        for cat in [q,f'{q}-merged',f'{q}-other-merged']:
            if cat in obj_cats:
                return [obj for obj in objs if obj['category']==cat]

        return None

    def execute(self,img,objs,query,category):
        query = query.split(',')
        select_objs = []

        if category is not None:
            cat_objs = [obj for obj in objs if obj['category'] in category]
            if len(cat_objs) > 0:
                objs = cat_objs


        if category is None:
            for q in query:
                matches = self.query_string_match(objs, q)
                if matches is None:
                    continue
                select_objs += matches

        if query is not None and len(select_objs) == 0:
            select_objs = API.select(query, objs, img)

        return select_objs


class ColorpopInterpreter():
    step_name = 'COLORPOP'

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def execute(self,img,objs):
        gimg = img.copy()
        gimg = gimg.convert('L').convert('RGB')
        gimg = np.array(gimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            gimg = mask*img + (1-mask)*gimg

        gimg = np.array(gimg).astype(np.uint8)
        gimg = Image.fromarray(gimg)

        return gimg


class BgBlurInterpreter():
    step_name = 'BGBLUR'

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def smoothen_mask(self,mask):
        mask = Image.fromarray(255*mask.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius = 5))
        return np.array(mask).astype(float)/255

    def execute(self,img,objs):
        bgimg = img.copy()
        bgimg = bgimg.filter(ImageFilter.GaussianBlur(radius = 2))
        bgimg = np.array(bgimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            mask = self.smoothen_mask(mask)
            bgimg = mask*img + (1-mask)*bgimg

        bgimg = np.array(bgimg).astype(np.uint8)
        bgimg = Image.fromarray(bgimg)

        return bgimg


class FaceDetInterpreter():
    step_name = 'FACEDET'

    def box_image(self,img,boxes):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            draw.rectangle(box,outline='blue',width=5)

        return img1

    def enlarge_face(self,box,W,H,f=1.5):
        x1,y1,x2,y2 = box
        w = int((f-1)*(x2-x1)/2)
        h = int((f-1)*(y2-y1)/2)
        x1 = max(0,x1-w)
        y1 = max(0,y1-h)
        x2 = min(W,x2+w)
        y2 = min(H,y2+h)
        return [x1,y1,x2,y2]

    def det_face(self,img):
        faces = API.face_detection(img)

        W,H = img.size
        objs = []
        for i,box in enumerate(faces):
            x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
            x1,y1,x2,y2 = self.enlarge_face([x1,y1,x2,y2],W,H)
            mask = np.zeros([H,W]).astype(float)
            mask[y1:y2,x1:x2] = 1.0
            objs.append(dict(
                box=[x1,y1,x2,y2],
                category='face',
                inst_id=i,
                mask = mask
            ))
        return objs

    def execute(self,image):
        objs = self.det_face(image)
        return objs


class EmojiInterpreter():
    step_name = 'EMOJI'

    def add_emoji(self,objs,emoji_name,img):
        W,H = img.size
        emojipth = os.path.join(EMOJI_DIR,f'smileys/{emoji_name}.png')
        for obj in objs:
            x1,y1,x2,y2 = obj['box']
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            s = (y2-y1)/1.5
            x_pos = (cx-0.5*s)/W
            y_pos = (cy-0.5*s)/H
            emoji_size = s/H
            emoji_aug = imaugs.OverlayEmoji(
                emoji_path=emojipth,
                emoji_size=emoji_size,
                x_pos=x_pos,
                y_pos=y_pos)
            img = emoji_aug(img)

        return img

    def execute(self,img,objs,emoji_name):
        img = self.add_emoji(objs, emoji_name, img)

        return img


class ListInterpreter():
    step_name = 'LIST'

    prompt_template = """
Create comma separated lists based on the query.

Query: List at most 3 primary colors separated by commas
List:
red, blue, green

Query: List at most 2 north american states separated by commas
List:
California, Washington

Query: List at most {list_max} {text} separated by commas
List:"""

    def get_list(self,text,list_max):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=self.prompt_template.format(list_max=list_max,text=text),
            temperature=0,
            max_tokens=256,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        )

        item_list = response.choices[0]['text'].lstrip('\n').rstrip('\n').split(', ')
        return item_list

    def execute(self,query,max):
        item_list = self.get_list(query,max)
        return item_list


class ClassifyInterpreter():
    step_name = 'CLASSIFY'

    def query_obj(self,query,objs,img):
        return API.query_obj(query,objs,img)

    def execute(self,img,objs,cats):
        import copy
        objs = self.query_obj(cats, copy.deepcopy(objs), img)

        return objs


class TagInterpreter():
    step_name = 'TAG'

    def tag_image(self,img,objs):
        W,H = img.size
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 24)
        # font = ImageFont.truetype(size=16)
        for i,obj in enumerate(objs):
            box = obj['box']
            draw.rectangle(box,outline='green',width=6)
            x1,y1,x2,y2 = box
            label = obj['class'] + '({})'.format(obj['class_score'])
            if 'class' in obj:
                w,h = font.getsize(label)
                if x1+w > W or y2+h > H:
                    draw.rectangle((x1, y2-h, x1 + w, y2), fill='green')
                    draw.text((x1,y2-h),label,fill='white',font=font)
                else:
                    draw.rectangle((x1, y2, x1 + w, y2 + h), fill='green')
                    draw.text((x1,y2),label,fill='white',font=font)
        return img1

    def execute(self,img,objs):
        img = self.tag_image(img, objs)

        return img


class ReplaceInterpreter():
    step_name = 'REPLACE'

    def create_mask_img(self,objs):
        mask = objs[0]['mask']
        mask[mask>0.5] = 255
        mask[mask<=0.5] = 0
        mask = mask.astype(np.uint8)
        return Image.fromarray(mask)

    def merge_images(self,old_img,new_img,mask):
        print(mask.size,old_img.size,new_img.size)

        mask = np.array(mask).astype(np.float)/255
        mask = np.tile(mask[:,:,np.newaxis],(1,1,3))
        img = mask*np.array(new_img) + (1-mask)*np.array(old_img)
        return Image.fromarray(img.astype(np.uint8))

    def resize_and_pad(self,img,size=(512,512)):
        new_img = Image.new(img.mode,size)
        thumbnail = img.copy()
        thumbnail.thumbnail(size)
        new_img.paste(thumbnail,(0,0))
        W,H = thumbnail.size
        return new_img, W, H

    def predict(self,img,mask,prompt):
        mask,_,_ = self.resize_and_pad(mask)
        init_img,W,H = self.resize_and_pad(img)
        new_img = API.replace(
            prompt=prompt,
            image=init_img,
            mask_image=mask,
            # strength=0.98,
            guidance_scale=7.5,
            num_inference_steps=50 #200
        )
        return new_img.crop((0,0,W-1,H-1)).resize(img.size)

    def execute(self,img,objs,prompt):
        
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        plt.imsave('filename.png', np.array(objs[0]["mask"]), cmap=cm.gray)
        print("Debuging")
        # box masking
        for idx, obj in enumerate(objs):
            mask_ = np.zeros((obj['mask'].shape))
            x1, y1, x2, y2 = obj['box']
            mask_[y1:y2,x1:x2] = 1
            objs[idx]['mask'] = mask_ 
        """
        #import pdb
        #pdb.set_trace()
        mask = self.create_mask_img(objs)
        new_img = self.predict(img, mask, prompt)
        return new_img


class DetectInterpreter():
    step_name = 'DETECT'

    def execute(self, image):
        boxes = API.object_detector(image)
        # boxes = API.glip(image,'object')
        selected_boxes = []
        image_size = image.size[0] * image.size[1]
        threshold = 0.01
        for box in boxes:
            if (box[2] - box[0]) * (box[3] - box[1]) > image_size * threshold:
                selected_boxes.append(box)
        return selected_boxes

class CAPTIONInterpreter():
    step_name = 'CAPTION'

    def execute(self, image):
        return API.blip(image)
