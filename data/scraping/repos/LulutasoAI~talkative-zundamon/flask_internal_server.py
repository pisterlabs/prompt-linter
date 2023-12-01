from flask import Flask, jsonify
from tts_handler import TTSHandler
from openai_functions import gptHandler
from flask import request
from flutter_information_capsule import FlutterInformationCapsule
from youtube_handler import youtubeHandler
from key import youtube_api_key as api_key
from key import video_id
from youtube_comments_handler import YoutubeCommentsHandler
from typing import Dict, List

class FlaskServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.gpt_handler = gptHandler()
        self.tts_handler = TTSHandler()
        self.server_status = "idle"  # idle, processing
        self.youtube_handler = youtubeHandler()
        self.youtube_comments_handler = YoutubeCommentsHandler()
        self.setup_routes()
        self.comments_to_process = []
        self.flutter_cupsules_to_send : list[FlutterInformationCapsule] = []

    def setup_routes(self):
        @self.app.route('/get_message', methods=['GET'])
        def get_message():
            #test use
            question = request.args.get('question')
            self.server_status = "processing"
            response = self.gpt_handler.callChatGPT("こんにちはなのだ")
            self.server_status = "idle"
            return jsonify({"message": response})
        
        @self.app.route('/fetch_comments', methods=['GET'])
        def fetch_comments():
            # Fetch live comments
            #call it every 5~10 seconds from flutter?
            try:
                live_comments_data = self.youtube_handler.fetch_live_comments(api_key, video_id)
                # Process comments (for now, just print them)
                comment_data_list = self.youtube_handler.extract_comment_data(live_comments_data)
                self.youtube_comments_handler.add_comments(comment_data_list)

                self.generate_flutter_information_cupsule()
                print(self.youtube_comments_handler.all_comments)
                return jsonify({"result": True})
            except Exception as e:
                print(e)
                return jsonify({"result": False})
        
        @self.app.route('/fetch_latest_info', methods=['GET'])
        def fetch_latest_info():
            #call it every 60 to 80 seconds from flutter?
            return self.send_flutter_information_cupsule()

    def send_flutter_information_cupsule(self):
        if len(self.flutter_cupsules_to_send) > 0:
            result = self.flutter_cupsules_to_send.pop(0)
            print(result.__json__())
            return jsonify(result.__json__())
        else:
            return jsonify({"result": False})
    
    def generate_flutter_information_cupsule(self):
        """
        youtube_handler = youtubeHandler()
        # Fetch live comments
        live_comments_data = youtube_handler.fetch_live_comments(api_key, video_id)
        comment_data_list = youtube_handler.extract_comment_data(live_comments_data)
        """
        """
        class FlutterInformationCapsule:
        def __init__(self, text, questioner, wav_path, emotion=None,):
        """
        new_comments:List[Dict] = self.youtube_comments_handler.get_new_comments()
        if len(new_comments) > 0:
            for comment in new_comments:
                question = comment["question"]
                
                questioner = comment["user"]
                text = self.gpt_handler.callChatGPT(question, questioner)
                image_url = comment["profile_picture_url"]
                print(image_url)
                wav_path = self.tts_handler.textToVoice(text)
                result = FlutterInformationCapsule(text, question, questioner, wav_path, image_url=image_url)
                self.flutter_cupsules_to_send.append(result)
        

    def run(self, debug=True):
        self.app.run(debug=debug)

if __name__ == '__main__':
    server = FlaskServer()
    server.run(debug=True)
