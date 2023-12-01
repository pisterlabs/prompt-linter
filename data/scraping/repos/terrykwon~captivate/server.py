import os

# from multiprocessing import Process, Lock, Barrier, Queue
from torch import multiprocessing
from torch.multiprocessing import Process, Queue

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from modelserver.unimodal import audial_infinite, visual
from modelserver.visualization.visualizer import Visualizer
from modelserver.guidance.guidance import Guidance

import time
import math

import numpy as np
import heapq


from collections import defaultdict

import cv2


import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/modelserver/default-demo-app-c4bc3-b67061a9c4b1.json"


guide_file_path = '/workspace/modelserver/modelserver/guidance/Guidance_sentences_0729.csv'

def start(url, queue, is_visualize):
    print('server start')
    server = ModelServer(url, queue)

    server.run(visualize=is_visualize)


class ModelServer:
    ''' The "server" is a main process that starts two additional processes: the
        'visual' process and the 'audial' process. Each process runs streaming
        inference on the decoded outputs of a video stream, which is done by
        starting decoder subprocesses.

        The inference outputs of the two processes are put into a `Queue`, where
        they are consumed by the main process.

        TODO:
          - Graceful shutdown. Currently one has to re-run the whole pipeline to
            start inference on a new video, which is very annoying. This can be
            done with some form of locking? Since the models have to be
            initialized in their respective thread.

          - Running the pipeline on a local video file and saving the output,
            for demo purposes.
    '''

    def __init__(self, stream_url, result_queue):
        # this method must be only called once
        # multiprocessing.set_start_method('spawn') # CUDA doesn't support fork
        # but doesn't matter if CUDA is initialized on a single process
        self.result_queue = result_queue
        
        self.stream_url = stream_url
        self.queue = Queue() # thread-safe

        ## for target weight decay (ms)
        self.curr_time = int(round(time.time() * 1000)) 

        
        self.visual_process = [ Process(target=visual.run, 
                args=(self.stream_url+str(camera_id), self.queue, None), daemon=True) for camera_id in range(1, 4) ]

        self.audial_process = Process(target=audial_infinite.run, 
                args=('rtmp://video:1935/captivate/test1', self.queue, None), daemon=True)

        # ## sync test
        # self.visual_process = [ Process(target=visual.run, args=('rtmp://video:1935/captivate/test', self.queue, None), daemon=True) ]
        # self.audial_process = Process(target=audial_infinite.run, args=('rtmp://video:1935/captivate/test', self.queue, None), daemon=True)
        # self.visualizer = [Visualizer(0)]

        self.visualizer = [ Visualizer(camera_id) for camera_id in range(1, 4) ]

        self.guidance = Guidance(guide_file_path)

        self.toys = self.guidance.get_toys()


        self.visual_classes = {
            'dog' : '강아지',
            'cat' : '고양이',
            'fish' : '물고기',
            'bear' : '곰돌이',
            'flower' : '꽃',
            'spoon' : '숟가락',
            'bicycle' : '자전거',
            'shoe' : '신발',
            'ball' : '공',
            'bus' : '버스',
            'bag' : '가방',
            'baby' : '아기',
        }
    

        
        print('Init modelserver done')


    def get_attended_objects(self, focus, object_bboxes, classes):
        ''' Returns the object at the `focus` coordinate.

            Epsilon widens the bbox by a fixed amount to accomodate slight
            errors in the gaze coordinates.

        '''
        attended_objects = []
        x, y = focus
        epsilon = 10 # widens the bounding box, pixels

        for i, bbox in enumerate(object_bboxes):
            # exclude person!
            if classes[i] == 'person':
                continue

            left, top, right, bottom = bbox
            if (x > left-epsilon and x < right+epsilon and 
               y < bottom+epsilon and y > top-epsilon):
                attended_objects.append(classes[i])

        return attended_objects
        

    def update_context(self, modality, targets):
        ''' Adds weight to the detected objects: by `alpha` for a single visual
            frame and by `beta` for an audial utterance.
        '''
        if modality == 'visual':
            alpha = 0.017 / 4   
        elif modality == 'audial':
            alpha = 0.1
        else:
            raise ValueError('modality must be one of visual or audial')
        

        ## get distribution
        target_length = 0
        target_dist = defaultdict(float)
        for t in targets:
            if t in self.guidance.toy_list:
                target_dist[t] += 1
                target_length += 1

        if target_length != 0:
            for toy in self.toys:
                toy.update_weight(target_dist,target_length,alpha)

        new_phrases = self.get_recommendations()

        return new_phrases

    def get_recommendations(self):
        ''' Returns a list of recommended words.

            The proportion of words are determined by the context weights.

            A max heap-like structure would be a lot more convenient than
            recalculating weights and sorting every time...
        '''
        N = 6 # Total number of words to recommend
        count = N
        
        recommendations = [] # list to order
        displayed_phrases = []


        for item in sorted(self.toys, key = lambda x: x.weight, reverse=True):
            obj = item.toy_name
            weight = item.weight
            
            # number of targets to recommend for this word
            n = math.ceil(weight * N)
            if count == 0:
                break
            elif count - n < 0:
                n = count
            count -= n
            
            heap_candidates = heapq.nlargest(int(n), item.phrases , key = lambda x : math.ceil(x.weight))
            
            
            for c in heap_candidates:

                recommendations.append(
                    {
                        'object' : obj,
                        'target_word' : c.word,
                        'target_sentence' : c.phrase,
                        'highlight' : c.highlight,
                        'id' : c.id,
                        'color' : c.color
                    }
                )


                displayed_phrases.append(c.phrase)
            
            for toy in self.toys:
                toy.set_display(displayed_phrases)
        
        recommendation_to_queue = {
            'tag' : 'recommendation',
            'recommendations' : recommendations
        }
        
        if self.result_queue:
            self.result_queue.put(recommendation_to_queue)

        return displayed_phrases

    
    def on_spoken(self, words, displayed_phrases):
        ''' Action for when a target word is spoken. TODO

            * This isn't the name of the object! It's the candidate.

            The word's relevance should be decreased a bit so that the parent
            diversifies words.
        '''
        target_spoken = []
        is_spoken_word = 0

        for word in words:
            for toy in self.toys:
                if toy.is_phrase_spoken(word, displayed_phrases):
                    is_spoken_word = 1

            if is_spoken_word:
                target_spoken.append(word)
                is_spoken_word = 0
            
        if len(target_spoken) > 0:
            
            target_to_queue = {
                'tag':'target_words',
                'words':target_spoken
            }

            if self.result_queue:
                self.result_queue.put(target_to_queue)
        return target_spoken
    
    def time_decay(self):
        curr_time = int(round(time.time() * 1000))
        
        is_phrase_ordered = 0
        for toy in self.toys:
            if toy.track_displayed_time(curr_time):
                is_phrase_ordered = 1
        
        return is_phrase_ordered

    def restart_audial_process(self):
        self.audial_process.join()

        self.audial_process = Process(target=audial_infinite.run, args=('rtmp://video:1935/captivate/test1', self.queue, None), daemon=True)
        self.audial_process.start()



    def run(self, visualize=False):
        ''' Main loop.
        '''
        # These processes should be joined on error, interrupt, etc.
        [ vp.start() for vp in self.visual_process ]

        self.audial_process.start()

        print('process start')
        # This is unnecessary because the queue.get() below is blocking anyways
        # self.barrier.wait()

        transcript = ''
        spoken_words_prev = []
        spoken_words_update = []
        target_spoken = []

        displayed_phrases = []

        image = None
        object_bboxes = []
        object_confidences = []
        object_classnames = []
        face_bboxes = []
        gaze_targets = []

        ## test for audio-video sync
        audio_time = ''
        video_time = ''


        

        ## init recommendation (first send)
        displayed_phrases = self.get_recommendations()


        while True:
            try:
                ## restart audio process when there is no signal
                if not self.audial_process.is_alive():
                    self.restart_audial_process()

                if self.time_decay(): ## True when there are re-ordered phrases
                    displayed_phrases= self.get_recommendations()
                

                # This blocks until an item is available
                result = self.queue.get(block=True, timeout=None) 

                if result['from'] == 'image':                    
                    image = result['image']
                    object_bboxes = result['object_bboxes']
                    object_confidences = result['object_confidences']
                    object_classnames = result['object_classnames']
                    face_bboxes = result['face_bboxes']
                    gaze_targets = result['gaze_targets']
                    camera_id = result['camera_id']-1
                    frame_num = result['frame_num']
                    video_time = result['video_time']


                    attended_objects = [] # includes both parent & child for now
                    for target in gaze_targets:
                        attended_objects.extend(self.get_attended_objects(
                                target, object_bboxes, object_classnames))
                    
                    target_objects = []
                    for o in attended_objects:
                        if o in self.visual_classes.keys():
                            object_korean = self.visual_classes[o]
                            target_objects.append(object_korean)
                    
                    # update if there's objects
                    if len(target_objects) != 0:
                        displayed_phrases = self.update_context('visual', target_objects)
                                        

                    if visualize:
                        
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        visualizer_curr = self.visualizer[camera_id]
                        visualizer_curr.draw_objects(image, object_bboxes, 
                                object_classnames, object_confidences)
                        visualizer_curr.draw_face_bboxes(image, face_bboxes)
                        for i, face_bbox in enumerate(face_bboxes):
                            visualizer_curr.draw_gaze(image, face_bbox, 
                                    gaze_targets[i])

                        #test for sync
                        image = visualizer_curr.add_captions_recommend(image,transcript,target_spoken)
                        visualizer_curr.visave(image, frame_num)
                    target_spoken.clear()



                elif result['from'] == 'audio':

                    transcript = result['transcript']

                    spoken_words_update = result['spoken_words_update']

                    audio_time = result['audio_time']
                    
                    print(transcript)


                    
                    # update spoken & target word weight
                    spoken = self.on_spoken(spoken_words_update, displayed_phrases)
                    if len(spoken) != 0:
                        target_spoken = spoken  


                    spoken_objects = []

                    for word in spoken_words_update:
                        if word in self.guidance.toy_list:
                            spoken_objects.append(word)
                    
                    if len(spoken_objects) != 0 :
                            displayed_phrases = self.update_context('audial', spoken_objects)
                    
                    # if transcript is final
                    # if result['is_final']: 
                    #     spoken_objects = []

                    #     # update spoken objects list
                    #     for word in spoken_words:
                    #         if word in self.guidance.toy_list:
                    #             spoken_objects.append(word)

                    #     # update object context
                    #     if len(spoken_objects) != 0 :
                    #         displayed_phrases = self.update_context('audial', spoken_objects)
                            
                        
                    
                        

            except Exception as excp:
                print(type(excp))
                print(excp.args)
                print(excp)
                ## close processes
                self.audial_process.terminate()
                [vp.terminate() for vp in self.visual_process]
                print("exit server run")      
                break  

    
