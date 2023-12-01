import cv2
import matplotlib.pyplot as plt

import gensim
import numpy as np
import spacy
import json
from time import time

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim
from pprint import pprint

import os, re, operator, warnings
warnings.simplefilter("ignore", DeprecationWarning)

from ctypes import *
import math
import random
import os
import json
from time import time

start = time()
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = False
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./darknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    #import cv2
    #custom_image = cv2.imread(image) # use: detect(,,imagePath,)
    #import scipy.misc
    #custom_image = scipy.misc.imread(image)
    #im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
    if debug: print("Loaded image")
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    if debug: print("did prediction")
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_image(im)
    if debug: print("freed image")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


netMain = None
metaMain = None
altNames = None

def performDetect(imagePath="data/tempo.jpg", thresh= 0.25, configPath = "./cfg/yolov3.cfg", weightPath = "yolov3.weights", metaPath= "./data/coco.data", showImage= True, makeImageOnly = False, initOnly= False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes requires libraries scikit-image and numpy

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

    thresh: float (default= 0.25)
        The detection threshold

    configPath: str
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        Path to the data file. Raises ValueError if not found

    showImage: bool (default= True)
        Compute (and show) bounding boxes. Changes return.

    makeImageOnly: bool (default= False)
        If showImage is True, this won't actually *show* the image, but will create the array and return it.

    initOnly: bool (default= False)
        Only initialize globals. Don't actually run a prediction.

    Returns
    ----------------------


    When showImage is False, list of tuples like
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
        The X and Y coordinates are from the center of the bounding box. Subtract half the width or height to get the lower corner.

    Otherwise, a dict with
        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
    """
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    if showImage:
        try:
            from skimage import io, draw
            import numpy as np
            image = io.imread(imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            imcaption = []
            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
            if not makeImageOnly:
                io.imshow(image)
                io.show()
            detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
        except Exception as e:
            print("Unable to show image: "+str(e))
    return detections

def get_tags_for_video(path,n_frames=20):
    video = cv2.VideoCapture('data/video.mp4')
    i=0
    keys = set()
    while (video.isOpened()):
        ret,frame = video.read()
        if ret:
            if(not i):
                cv2.imwrite('data/tempo.jpg',frame)
                data = performDetect(showImage= False, makeImageOnly = False, initOnly= False)
                for key in data:
                    keys.add(key[0])
            i = (i+1)%n_frames
        else :
            break
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    video.release()
    key_tags = []
    for key in keys:
        key_tags.append(key)
    return {'video_key_tags':key_tags}


def clean(text):
    return str(text)

def build_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname: File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    yield gensim.utils.simple_preprocess(fname, deacc=True, min_len=3)
    
def process_speech_to_text(path,file_format):
    import speech_recognition as sr
    from pydub import AudioSegment

    sound = AudioSegment.from_file(str(path), format=str(file_format))
    r = sr.Recognizer()
    article_list = []

    for i in range(0,int(len(sound)/1000),15):
        if(i+15<int(len(sound)/1000)):
#             print(i)
            cropped = sound[i*1000:(14+i)*1000]
            cropped.export("file.wav", format="wav")
            with sr.AudioFile('file.wav') as source:
                audio = r.record(source)
            try:
                text = r.recognize_google(audio)
    #             article.join(' '+str(text))
                article_list.append(text)
            except :
                text = r.recognize_sphinx(audio)
                article_list.append(text)
    #             article.join(' '+str(text))

    cropped = sound[i*1000:int(len(sound))]
    cropped.export("file.wav", format="wav")
    with sr.AudioFile('file.wav') as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
    #             article.join(' '+str(text))
        article_list.append(text)
    except :
    #     text = r.recognize_sphinx(audio)
    #     article_list.append(text)
        print('Error while converting text: Please check your internet connection and Google Recognizer API installation. If you have installed Google Recognizer, check for the installation of Sphinx Recognizer.')
    string = ''
    for crop in article_list:
        string=string+' '+crop
    return string

def ret_top_model(threshold,corpus,dictionary,texts):
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < threshold:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics

def get_tags_for_audio(data,threshold=0.5):
    nlp = spacy.load("en_core_web_lg")
    for word in nlp.Defaults.stop_words:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True
    explicit = dict({'cum dumpster':82, 'felch':82, 'cunt':82, 'skullfuck':82, 'Alabama hot pocket':82, 'cock-juggling thundercunt':82,'rusty trombone': 82,'blumpkin':82,'Cleveland S-teamer':82,'cum guzzling cock sucker':81,'glass bottom boat':81,"suck a fat baby's dick":81,'skermit':80,'fucking pussy':80,'meat flap':80,'fuck hole':80,'hairy axe wound':79,'up the ass':79, 'assmucus':79,'cumdump':79, 'beef curtain':79, 'moose nuckle':79,'cum chugger':78,'mother fucker':78, 'motherfucking':78, 'roast beef curtains':78, 'fuck':78, 'Roman Helmet':78, 'dick':78,'get some squish':77, 'eat a dick':77, 'clitty litter':77, 'eat hair pie':77, 'bisnotch':77, 'yard cunt punt':77, 'blue waffle':77, 'fist fuck':77, 'bitchass mother fucker':77,'fuck me in the ass with no Vaseline':77,'fuck yo mama':77,'chota bags':77, 'cuntee':77, 'motherfucker':77, 'meat drapes':77,'schlong juice':76, 'bang':76, 'meat tulips':76,     'cum freak':76, 'buggery':76, 'cuntsicle':76,     'fuckmeat':76, 'bust a load':76, 'butt fuck':76, 'GMILF':76, 'cock snot':76, 'shit fucker':76, 'sausage queen':76, 'fucktoy':76, 'dick hole':76, 'cock pocket':76, 'lick my froth':76, 'cunt-struck':76, 'cockbag':76,  'gangbang':75, 'pussy fart':75, 'ham flap':75, 'cum guzzler':75, 'squeeze a steamer':75, 'ass fuck':75, 'hoitch':75, 'cunt hole':75, 'clit licker':75, 'anal impaler':75, 'dick sucker':75, 'baby arm':75, 'smoke a sausage':75, 'Cuntasaurus rex':75, 'cunt face':75, 'buckle buffer':75,     'slich':75, 'fubugly':   75,     'man chowder':  75,     'key hole':  75,     'cocksucker':  75,     'get redwings':  75,     'hemped up':  75,     'smoke pole' :  75,     'like fuck' : 75,     'feedbag material':  75,     'eat fur pie':  74,     'analconda': 74 ,    'soggy muffin' : 74,     'suck a dick' : 74, 'nut butter':   74 ,    'fuck-bitch':  74 ,    "pull (one's) dick" :  74,     'get brain':  74  ,   'sweet dick daddy with the candy balls' : 74, 'get in pants':  74  ,   'felcher':  74  ,   'fuck puppet' : 74})
    doc = nlp(clean(data))
    texts, article = [], []
    explicit_content=[]
    explicit_score=0
    explicit_score_i=0
    avg_score=0
    for w in doc:
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and not (w.text in explicit.keys()):
            article.append(w.lemma_)
        if w.text == '\n':
            texts.append(article)
            article = []
    for bad in explicit.keys():
        if bad in str(doc):
            explicit_score+=explicit[bad]
            explicit_score_i += 1
    if explicit_score_i:
        avg_score = explicit_score/explicit_score_i 
    if not len(texts):
        articles = []
        phrases = gensim.models.phrases.Phrases(article)
        bigram = gensim.models.phrases.Phraser(phrases)
        articles.append(bigram[article])
        dictionary = Dictionary(articles)
        corpus = [dictionary.doc2bow(text) for text in articles]
        texts = articles
    else:
        phrases = gensim.models.phrases.Phrases(texts)
        bigram = gensim.models.phrases.Phraser(phrases)
        texts = [bigram[line] for line in texts]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
    train_texts = list(build_texts(data))
    lm, top_topics = ret_top_model(threshold,corpus,dictionary,texts)
    lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]
    topics=set()
    for topic in lda_lsi_topics:
        for string in topic:
            topics.add(string)
    key_tags = []
    for key in topics:
        key_tags.append(key)
    for e in doc.ents:
        if e.text.lower() not in key_tags:
            key_tags.append(e.text.lower())
    return list([{'audio_key_tags':key_tags},{'explicit_content_score':avg_score}])

def create_json_file(filepath,aud_tags,vid_tags):
    final = [vid_tags,aud_tags]
    with open(filepath, 'w') as outfile:
        json.dump({'tags':final}, outfile,indent=4)
    
#vid_tags = get_tags_for_video('data/video.mp4')
#data = process_speech_to_text("data/video.mp4","mp4")
#aud_tags = get_tags_for_audio(data)
#create_json_file('key_tags.json',aud_tags,vid_tags)
