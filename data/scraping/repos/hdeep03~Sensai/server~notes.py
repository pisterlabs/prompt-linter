import openai
import os
from utils import load_transcript
import re
from multiprocessing import Pool
from tqdm import tqdm
import keras_ocr
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from PIL import Image as PILImage
import shutil
from pytube import YouTube
import pytube
import time
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import utils
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import cv2
import torch
import gc
import numpy as np
import requests
USING_GPU = torch.cuda.is_available()


##--Downloading and getting stuff from video

def download_youtube(vidlink,outlocation):
  try:
    yt = YouTube(vidlink)
  except:
    print("Connection Error") 
  
  mp4files = yt.streams.filter(file_extension="mp4")
  d_video = mp4files[0]
  if os.path.exists(outlocation):
    shutil.rmtree(outlocation)
  try:
    d_video.download(outlocation)
  except:
    print("Some Error!")
  video_name = os.listdir(outlocation)[0]
  filep = f"{outlocation}/{os.listdir(outlocation)[0]}"
  os.rename(filep,f"{outlocation}/video.mp4")
  return video_name

def extract_every_n_frames_from_video(video_path,every,outfolder):
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    os.makedirs(outfolder)
    os.system(f"ffmpeg -i \"{video_path}\" -vf \"select=not(mod(n\,{every}))\" -vsync vfr {outfolder}/img_%03d.jpg")

def apply_func_to_frames(frames_path, func, start=-1, end=-1):                     
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(os.listdir(frames_path))

    frames_list = list(range(start, end))
    outlist = []

    for n,file in enumerate(os.listdir(frames_path)):
        if n in frames_list:
            outlist.append(func(cv2.cvtColor(np.array(PILImage.open(f"{frames_path}/{file}")), cv2.COLOR_RGB2BGR)))
    return outlist

def get_vid_infos(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        "fps":fps,
        "length":frame_count
    }


##--Identifying keyframes

def get_keyframes(frames_path, start=-1, end=-1, every=300, remove_dup_screens=True):
    DUPLICATE_SCREENS_IOU_THRESH = 0.65 ##threshold for determining whether two screens are the same screen
    CHARACTER_COUNT_THRESHOLD = 4 ##frames with less than this many detected characters will be discarded
    HUMAN_INCLUSION_THRESH = 0.1 ##no more than this percentage of the frame can be taken up by a human
    NO_HUMAN_BOX = (0.2,0.8,0.1,0.9) ##x1,x2,y1,y2; range of the dimensions of the image where there should absolutely be no humans
    TEXT_ACCEPTABLE_BOX = (0.3,0.7,0.1,0.9) ##x1,x2,y1,y2; box within which the center of mass of text must lie

    frames_smallframes = apply_func_to_frames(frames_path, lambda x:[cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
                                                        cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), (0,0), fx = 0.7, fy = 0.7)])

    frames = [x[0] for x in frames_smallframes]
    low_def_frames = [x[1] for x in frames_smallframes]
    print("asdf",len(frames))

    if USING_GPU:
        torch.cuda.set_per_process_memory_fraction(float(1), device=None)

    def download_link(url,outp):
        print(f"downloading {url}...")
        downloaded_obj = requests.get(url)

        with open(outp, "wb") as file:
            file.write(downloaded_obj.content)

    if not os.path.exists("public/assets/pointrend_resnet50.pkl"):
        if not os.path.exists("public/assets"):
            os.makedirs("public/assets")
        download_link("https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl",
                        "public/assets/pointrend_resnet50.pkl")

    ins = instanceSegmentation()
    ins.load_model("public/assets/pointrend_resnet50.pkl", confidence = 0.2, detection_speed = "rapid")

    def frame_2_content_amt(frame):  
        mask_off_objects = ["person"]

        blur_size = 5
        dilation_size = 5

        #PILImage.fromarray(frame.astype('uint8'), 'RGB').save(f"frame_images/frame_1.png")
        cv2.imwrite("frame_1.jpg",frame)
        results = ins.segmentImage(f"frame_1.jpg")[0]#person_mask_results[person_mask_ind["value"]]
        person_mask = np.zeros(frame.shape[:-1])
        masks = results["masks"]
        for n,class_name in enumerate(results["class_names"]):
          if class_name in mask_off_objects:
            person_mask += masks[:,:,n]
        del results
        dilate_kernel = np.ones((dilation_size,dilation_size), np.uint8)
        dilated_person_mask = cv2.dilate((person_mask>0).astype(np.uint8), dilate_kernel, iterations=1)

        no_human_zone_mask = np.zeros(frame.shape[:-1])
        width,height = frame.shape[:-1]
        no_human_zone_mask[int(width*NO_HUMAN_BOX[0]):int(width*NO_HUMAN_BOX[1]),
                           int(height*NO_HUMAN_BOX[2]):int(height*NO_HUMAN_BOX[3])]
        if np.sum(person_mask) > frame.shape[0]*frame.shape[1]*HUMAN_INCLUSION_THRESH:
          return None
        if np.sum(no_human_zone_mask*person_mask) > 0:
          return None
        person_mask_out = 1-dilated_person_mask
        no_person_frame = frame*np.stack([person_mask_out,person_mask_out,person_mask_out],axis=-1)
        gray = cv2.cvtColor(no_person_frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,5)

        frame_blur = cv2.GaussianBlur(no_person_frame.astype(np.uint8), (blur_size,blur_size), 0) 
        edges = cv2.Canny(image=frame_blur, threshold1=100, threshold2=200)
        edges_bin = (edges/np.max(edges) > 0.05).astype(np.float32)

        dilate_kernel = np.ones((dilation_size,dilation_size), np.uint8)

        dilated_person_mask = cv2.dilate(person_mask, dilate_kernel, iterations=2)
        edges_bin = (np.ones(dilated_person_mask.shape) - dilated_person_mask)*edges_bin

        img_dilation = cv2.dilate(edges_bin, dilate_kernel, iterations=4)
        return img_dilation

    
    ##get frame morphological content and frames
    frame_contents = apply_func_to_frames(frames_path, frame_2_content_amt)

    if USING_GPU:
        torch.cuda.set_per_process_memory_fraction(float(0), device=None)
        gc.collect()
        torch.cuda.empty_cache()

    def zip_and_filter_lists(lists,filter_func,ind=0):
      newlists = []
      for lst in lists:
        newlists.append([])
      for n,x in enumerate(lists[ind]):
        if filter_func(x):
          for j in range(len(lists)):
            newlists[j].append(lists[j][n])
      return newlists
    
    frame_contents,frames,low_def_frames,frame_poss = zip_and_filter_lists([frame_contents,frames,low_def_frames,range(len(frame_contents))],lambda x:not x is None)

    def calc_iou(a,b):
      a = (a > 0).astype(np.uint8)
      b = (b > 0).astype(np.uint8)
      return np.sum(a+b==2)/np.sum((a+b) > 0)


    ##generate differences between frames based on morphological content
    ious = []
    for n,frame_content in enumerate(frame_contents):
      frame_content = (frame_content > 0).astype(np.uint8)
      if n==0: continue
      iou = calc_iou(frame_content,frame_contents[n-1])
      ious.append(1-iou)

    print("before morph len",len(frame_contents))


    ## filter by morphological content amt
    morph_key_frames = []
    for n,iou in enumerate(tqdm(ious,"performing duplicate/similar frame filtering using morphological content")):
      if iou <= DUPLICATE_SCREENS_IOU_THRESH:
        morph_key_frames.append((n,frame_contents[n]))
        morph_key_frames.append((n+1,frame_contents[n+1]))
    
    print("after morph len",len(morph_key_frames))

    def perform_predict_batch(items,predict_func,batch_size):
      def split_batches(items,size):
        batches = []
        for i in range(0,(len(items)//size)+1):
          shard = items[i*size:(i+1)*size]
          if len(shard) > 0:
            batches.append(shard)
        return batches
      all_preds = []
      batches = split_batches(items,batch_size)
      for n,batch in enumerate(tqdm(batches,f"predicting {len(batches)} batches for {len(items)} items")):
        all_preds.extend(predict_func(batch))
      return all_preds
    
    ##create predictions for strings for OCR
    pipeline = keras_ocr.pipeline.Pipeline()
    frames_2_predict = [low_def_frames[frame_num] for frame_num,_ in morph_key_frames]
    frame_string_predictions = perform_predict_batch(frames_2_predict,pipeline.recognize,2)
    
    ##filter by OCR character count and text "center of mass"
    height,width = frames_2_predict[0].shape[:-1]
    OCR_key_frame_contents = []
    for frame_string_pred, (frame_num,keyframe_content) in zip(frame_string_predictions, tqdm(morph_key_frames,"performing filtering using OCR")):
      def parse_string_pred(string_pred):
        out_string = []
        for pred in string_pred:
          out_string.append(pred[0])
        return " ".join(out_string)

      def calc_bbs_cm(bbs): ##DEBUG THIS
        total_mass = 0
        total_mass_cms = np.array([0.0,0.0])
        for bb in bbs:
          x1,x2,y1,y2 = bb
          bb_cm = np.array([x1+(x2-x1)//2,y1+(y2-y1)//2])
          bb_mass = (x2-x1)*(y2-y1)
          total_mass+=bb_mass
          total_mass_cms+=bb_cm*bb_mass
        final_cm = total_mass_cms/total_mass
        return (final_cm[0],final_cm[1])
      
      def calc_image_text_cm(string_pred):
        bbs = []
        for pred in string_pred:
          x1 = np.min(pred[1][:,0],axis=0)
          x2 = np.max(pred[1][:,0],axis=0)
          y1 = np.min(pred[1][:,1],axis=0)
          y2 = np.max(pred[1][:,1],axis=0)

          bbs.append((x1,x2,y1,y2))
        cm = calc_bbs_cm(bbs)
        return cm
      
      bbs_cm = calc_image_text_cm(frame_string_pred)

      ##filter by center of mass location
      cm_x,cm_y = bbs_cm
      if not (width*TEXT_ACCEPTABLE_BOX[0] <= cm_x <= width*TEXT_ACCEPTABLE_BOX[1] and \
              height*TEXT_ACCEPTABLE_BOX[2] <= cm_y <= height*TEXT_ACCEPTABLE_BOX[3]):
        continue

      frame_string = parse_string_pred(frame_string_pred)
      chars = [x for x in frame_string.split() if x]
      if len(chars) > CHARACTER_COUNT_THRESHOLD:
          OCR_key_frame_contents.append((frame_num,keyframe_content))
          

    print("OCR length",len(OCR_key_frame_contents))

    if remove_dup_screens:
      ##remove duplicate keyframes based on IOU
      non_dup_frames = []
      OCR_key_frame_contents.reverse()
      for frame_num,OCR_key_frame_content in OCR_key_frame_contents:
        if len(non_dup_frames) == 0 or calc_iou(OCR_key_frame_content,non_dup_frames[-1][1]) <= DUPLICATE_SCREENS_IOU_THRESH:
          non_dup_frames.append((frame_num,OCR_key_frame_content))
      non_dup_frames.reverse()
    else:
      non_dup_frames = OCR_key_frame_contents


    ##get the original color frames
    output_frames = []
    for frame_num,output_frame in non_dup_frames:
      output_frames.append((frame_poss[frame_num],frames[frame_num]))
    return output_frames

def delete_time_proximal_frames(video_keyframes, video_path):
    video_infos = get_vid_infos(video_path)
    video_fps,video_length = video_infos["fps"],video_infos["length"]
    video_length = video_length/video_fps
    video_keyframes = sorted(video_keyframes,key = lambda x:x[0])

    non_dup_keyframes = []
    video_keyframes.reverse()
    for keyframe_pos,keyframe in video_keyframes:
        keyframe_timestamp_secs = keyframe_pos*READ_FRAME_INTERVAL/video_fps
        if len(non_dup_keyframes) > 0:
            if non_dup_keyframes[-1][0]-keyframe_timestamp_secs >= MINIMUM_TIME_DIFFS_SECS:
                print(keyframe_timestamp_secs)
                non_dup_keyframes.append((keyframe_timestamp_secs,keyframe))
        else:
            print(keyframe_timestamp_secs)
            non_dup_keyframes.append((keyframe_timestamp_secs,keyframe))
    non_dup_keyframes.reverse()
    return non_dup_keyframes

##--Generating notes based on transcript

def parse_transcript_dict(transcript_dict, read_range = None, only_text=True, preserve_sentences=True):
  ##preserve_sentences could potentially cause problems if no sentence structure is present within the transcript

  def convert(transcript):
    starts = transcript["starts"]
    texts = transcript["text"]

    new_dict = []
    for start,text in zip(starts,texts):
        new_dict.append({"start":start,"text":text})
    return new_dict

  transcript_dict = convert(transcript_dict)

  if read_range is None:
    read_range = (0,transcript_dict[-1]["start"])

  text = []
  range_start = False
  range_end = False
  for t,trancsript_item in enumerate(transcript_dict):
      item_starttime = trancsript_item["start"]

      if preserve_sentences:
        ##starting reading; ensure sentence start
        if item_starttime >= read_range[0] and not range_start:
          x=1
          while t-x >= 0:
            if transcript_dict[t-x]["text"].strip().endswith("."):
              x-= 1
              text.extend([trancsript_item["text"] for trancsript_item in transcript_dict[t-x:t]])
              break
            x+=1
          range_start = True

        ##ending reading; ensure sentence end
        elif item_starttime > read_range[1] and not range_end:
          x=-1
          while t+x < len(transcript_dict):
            if transcript_dict[t+x]["text"].strip().endswith("."):
              text.extend([trancsript_item["text"] for trancsript_item in transcript_dict[t:t+x+1]])
              break
            x+=1
          range_end = True

      if read_range[0] <= trancsript_item["start"] <= read_range[1]:
        text_items = trancsript_item["text"].split()
        text.extend(text_items)

  text = " ".join(text)
  if only_text:
    text = re.sub("\[[\s\w]+\]","",text)
  text_segments = text.split()
  text = [seg for seg in text_segments if seg]
  return " ".join(text)

def create_text_shards(text, shardsize, sliding_amt):
  text_shards = []
  for i in range(len(text)//sliding_amt+1):
    def find_from_ind(string,text,ind,direction):
      while True:
        if ind >= len(string):
          return ind
        if ind <= 0:
          return 0
        ind_char = string[ind]
        if ind_char == text:
          return ind+direction
        ind+=direction
    start_ind = find_from_ind(text," ",i*sliding_amt,1)
    end_ind = find_from_ind(text," ",i*sliding_amt+shardsize,-1)+1
    text_shards.append(text[start_ind:end_ind])
    
  return text_shards

def run_davinci_notes(text):
  trancsript_shards = create_text_shards(text,3000,2500)

  output_lines = []
  for shard in tqdm(trancsript_shards):
    print("shard:",shard)
    while True:
      try:
        davinci3 = openai.Completion.create(model="text-davinci-003", 
                                            prompt=f"take notes on the following sentences using '*-' as a bullet point: \n\n{shard}",
                                            max_tokens=3000)
        break
      except Exception as e:
        print("error:",e)
        pass
    shard_notes = davinci3.choices[0].text

    ##Soemtimes davinci tries to create a title; only take lines that are actual bullet points
    for line in shard_notes.split("\n"):
      if re.match("^\*-\s?\w.+$",line):
        output_lines.append(re.sub("\*-\s?","- ",line))
      else:
        print("discarded:",line)
  return "\n".join(output_lines)

def generate_notes_for_keyframes(keyframes, video_transcript, video_length):
    ##create intervals for notetaking
    intervals = []
    for n,(keyframe_timestamp_secs,keyframe) in enumerate(keyframes):
        if n==0:
            intervals.append(parse_transcript_dict(video_transcript,read_range=(0,keyframe_timestamp_secs)))
        elif n < len(keyframes)-1:
            intervals.append(parse_transcript_dict(video_transcript,read_range=(keyframes[n-1][0],keyframe_timestamp_secs)))
        else:
            intervals.append(parse_transcript_dict(video_transcript,read_range=(keyframes[n-1][0],video_length+1)))

    ##generate notes for each interval
    def get_notes(n_and_interval):
        n,interval = n_and_interval
        notes = run_davinci_notes(interval)
        if len(notes.split("\n")) >= MINIMUM_NOTES_LINES_THRESHOLD:
            return (n,notes)
        else:
            return None

    import time
    start_time = time.time()
    ##parallel querying for notes
    #p = Pool(10)
    interval_notes = [get_notes(x) for x in  [(n,interval) for n, interval in enumerate(intervals)]]
    interval_notes = [note for note in interval_notes if note is not None]
    print("finished querying for notes in",time.time()-start_time)
    return interval_notes


def generate_pdf_from_interval_notes(interval_notes, non_dup_keyframes, outp, VIDEO_NAME, pdf_params):
    IMAGE_SIZE_INCHES,IMAGE_SIZE_LARGE,TEXT_WIDTH_CHARS,IMAGE_LINES_LIMIT,MAX_LINES_LIMIT = pdf_params
    title_style = ParagraphStyle(
    'title',
        alignment=TA_CENTER,
        fontSize=20,
        fontName="Times-Bold",
        leading=24
    )
    subtitle_style = ParagraphStyle(
    'subtitle',
        alignment=TA_CENTER,
        fontSize=14,
        fontName="Times-Bold"
    )
    text_style = ParagraphStyle(
        'default',
        fontName="Times-Roman",
        fontSize=12,
    )
    chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('FONTSIZE', (0, 0), (-1, -1), 12),
                            ('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'),],
    )
    doc = SimpleDocTemplate(outp,pagesize=letter,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=18)

    def get_image_preserve_ratio(path, width=1*inch):
        img = utils.ImageReader(path)
        iw, ih = img.getSize()
        aspect = ih / float(iw)
        return Image(path, width=width, height=(width * aspect))

    def trim_string_to_text_width(string,width):
        def split_to_width(string,width):
            words = string.split()

            current_segment = []
            segments = []
            for word in words:
                if len(" ".join(current_segment))+1+len(word) > width:
                    segments.append(" ".join(current_segment))
                    current_segment = []
                current_segment.append(word)
            segments.append(" ".join(current_segment))
            return segments
        output_lines = []
        for line in string.split("\n"):
            ##check if bulletpoint
            if line.startswith("-") or line.startswith("*") or line.startswith("."):
                split_lines = split_to_width(line,width-2)
                output_lines.extend([("  " if n > 0 else "") + split_line for n,split_line in enumerate(split_lines)])
            else:
                output_lines.extend(split_to_width(line,width))
        return "\n".join(output_lines)

    def detect_note_len(interval_note):
        interval_note_lines = interval_note.split("\n")

        def reconsolidate_lines(lines):
            new_lines = [""]
            for line in lines:
                if line.startswith("-") or line.startswith("*") or line.startswith("."): ##bulletpoint
                    new_lines.append(line)
                else:
                    new_lines[-1]+=" " + line
            return "\n".join(new_lines)
        def split_bullet_points_greater_than_length(lines,length):
            shards = [[],[]]
            shard_mode = 0
            for l,line in enumerate(lines):
                if l+1>length and (line.startswith("-") or line.startswith("*") or line.startswith(".")):
                    shard_mode = 1
                shards[shard_mode].append(line)
            return ["\n".join(shard) for shard in shards]

        if len(interval_note_lines) <= IMAGE_LINES_LIMIT:
            return [interval_note],"h"
        elif len(interval_note_lines) <= MAX_LINES_LIMIT:
            nexto,after = split_bullet_points_greater_than_length(interval_note,IMAGE_LINES_LIMIT)
            return [nexto,reconsolidate_lines(after.split("\n"))],"h"
        else:
            return [reconsolidate_lines(interval_note_lines)],"v"


    Story=[]
    if os.path.exists("imgtmps"):
        shutil.rmtree("imgtmps")
    os.mkdir("imgtmps")

    Story.append(Story.append(Paragraph(f"Notes for \"{VIDEO_NAME}\"", title_style)))
    Story.append(Spacer(1, 18))
    Story.append(Story.append(Paragraph(f"(Generated by SensAI)", subtitle_style)))
    Story.append(Spacer(1, 48))
    for interval_num, interval_note in interval_notes:
        print(interval_note)
        print("\n\n")
        interval_note = trim_string_to_text_width(interval_note,TEXT_WIDTH_CHARS)
        print(interval_note)
        
        interval_keyfram_im = PILImage.fromarray(non_dup_keyframes[interval_num][1].astype('uint8'), 'RGB')
        interval_keyfram_im.save(f"imgtmps/image_{interval_num}.png")
        keyframe_im = get_image_preserve_ratio(f"imgtmps/image_{interval_num}.png",width=round(IMAGE_SIZE_INCHES*inch))
        
        table_content = [interval_note, keyframe_im]
        if interval_num%2==1:
                table_content.reverse()
        Story.append(Table([table_content],style=chart_style))
        Story.append(Spacer(1, 12))

    doc.build(Story)
    shutil.rmtree("imgtmps")

openai.api_key = os.environ["OPENAI_API_KEY"]

##General settings
CACHE_PATH = './cache/'
VIDEO_ID = "jANZxzetPaQ&ab"
READ_FRAME_INTERVAL = 1500
MINIMUM_TIME_DIFFS_SECS = 30 ##limit time between keyframes to at least this amount of time
MINIMUM_NOTES_LINES_THRESHOLD = 3 ##if the generated notes are less than this many lines, discard this keyframe

##PDF settings
IMAGE_SIZE_INCHES = 4
IMAGE_SIZE_LARGE = 8
TEXT_WIDTH_CHARS = 40
IMAGE_LINES_LIMIT = 14 ##If notes are longer than this many lines, we extend out the rest of the notes across the screen
MAX_LINES_LIMIT = 30 ##If notes are longer than this many lines, then we will place the image then text vertically

def create_notes_pdf(video_id):
    VIDEO_NAME = download_youtube(f"https://www.youtube.com/watch?v={video_id}","videop").split(".")[0]
    video_infos = get_vid_infos("videop/video.mp4")
    video_fps,video_length = video_infos["fps"],video_infos["length"]
    video_length = video_length/video_fps

    extract_every_n_frames_from_video(f"videop/video.mp4",READ_FRAME_INTERVAL,"frames")
    video_keyframes = get_keyframes("frames")
    video_keyframes = delete_time_proximal_frames(video_keyframes,"videop/video.mp4")

    transcript = load_transcript(video_id)
    print(transcript)

    video_keyframes = sorted(video_keyframes,key = lambda x:x[0])
    interval_notes = generate_notes_for_keyframes(video_keyframes,transcript,video_length)
    print(interval_notes)
    print("gottemgottem")
    generate_pdf_from_interval_notes(interval_notes,video_keyframes,"static/Notes.pdf", VIDEO_NAME, [IMAGE_SIZE_INCHES,IMAGE_SIZE_LARGE,TEXT_WIDTH_CHARS,IMAGE_LINES_LIMIT,MAX_LINES_LIMIT])
    return "static/Notes.pdf" 

if __name__ == "__main__":
    print(create_notes_pdf(VIDEO_ID))
