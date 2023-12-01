import logging
import threading
from speech_recognition import Recognizer, Microphone
import speech_recognition as sr
import openai

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-7s : %(message)s\n')

class Google_STT:
    def __init__(self, lang="ko-KR"):
        self.r = Recognizer()
        self.mic = Microphone()
        self.trigger_words = ["이루", "루머", "헬로", "Hello", "안녕"]
        self.lang = lang

    def setup_mic(self):
        with self.mic as source:
            logging.info("주변 소음을 측정합니다.")
            self.r.adjust_for_ambient_noise(source)
            logging.info("소음 측정 완료.")

    def listen(self, stop_listening_event):
        while not stop_listening_event.is_set():
            try:
                with self.mic as source:
                    logging.info("호출 명령어를 탐지하는 중...")
                    audio = self.r.listen(source, timeout=4, phrase_time_limit=2)
                self.audio_queue.append(audio)
            except sr.WaitTimeoutError:
                pass  # 타임아웃 발생 시 무시하고 계속 진행

    def recognize(self, stop_listening_event):
        while not stop_listening_event.is_set():
            if self.audio_queue:
                audio = self.audio_queue.pop(0)
                try:
                    text = self.r.recognize_google(audio, language=self.lang)
                    logging.info("[수집된 음성]: {}".format(text))
                    if any(trigger_word in text for trigger_word in self.trigger_words):
                        stop_listening_event.set()
                        self.triggered = text
                except sr.UnknownValueError:
                    logging.info("호출 명령어가 없습니다.")
                except Exception as e:
                    logging.error("Error: {}".format(e))

    def listen_for_trigger(self):
        self.audio_queue = []
        self.triggered = None
        stop_listening_event = threading.Event()
        listen_thread = threading.Thread(target=self.listen, args=(stop_listening_event,))
        recognize_thread = threading.Thread(target=self.recognize, args=(stop_listening_event,))
        
        listen_thread.start()
        recognize_thread.start()
        
        listen_thread.join()
        recognize_thread.join()

        return self.triggered is not None

    def listen_for_task(self):
        try:
            with self.mic as source:
                logging.info("Task을 듣는 중...")
                audio = self.r.listen(source, timeout=6, phrase_time_limit=5)
            task_text = self.r.recognize_google(audio, language=self.lang)
            logging.info("[수집된 Task 내용]: {}".format(task_text))
            return task_text
        
        except sr.UnknownValueError:
            logging.info("음성 결과를 얻지 못했습니다.")
            return None
        
        except Exception as e:
            logging.error("Error: {}".format(e))
            return False


class STT_Agent(Google_STT):
    def __init__(self, lang="ko-KR"):
        super().__init__(lang)
        self.destination_list = ['미래관', '정문', '후문', '본관', '학생회관', '시대융합관', '창공관']
        self.PROMPT_FOR_SYSTEM = f"한국어로 대화합니다. 당신의 역할은 사용자의 질문에 대해 대답을 생성하는 것이 아닌 사용자의 질문을 5가지 경우로 필터링하는 것입니다.\
                                  1. 사용자의 질문이 {self.destination_list} 중 어디로 가고 싶다는 내용이면 'M:(리스트의 해당 목적지 index 번호)'로 대답해주세요. \
                                  2. 사용자가 빨리 가라고 요청하면 'SF'라고만 대답해주세요.\
                                  3. 사용자가 천천히 가라고 요청하면 'SL'라고만 대답해주세요.\
                                  4. 그 외의 모든 경우, 사용자가 물어본 내용을 앵무새처럼 'Q:사용자의 질문' 형태로 똑같이 대답해주세요.\
                                  5. 사용자가 춤추라고 하면 'D'라고 대답해주세요.\
                                  'Q:사용자의 질문'은 서울시립대학교 캠퍼스 홍보 LLM 모델에 질문으로 입력되어 대답을 생성하는데 사용됩니다.\
                                  만약 학교와 연관성이 전혀 없는 질문이거나 도덕적으로 부적절한 질문이라면, 'No'라고 대답해주세요. "

        
    def process_user_input(self, messages, model="gpt-3.5-turbo", temp=0):
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temp,
            messages=messages
            )['choices'][0]['message']['content']
        return response

    def filtering_task(self, predicted_text):
        messages = [
            {"role": "system", "content": self.PROMPT_FOR_SYSTEM},
            {"role": "user", "content": "동아리는 뭐가 있어?"},
            {"role": "assistant", "content": "Q:동아리는 뭐가 있어?"},
            {"role": "user", "content": "미래관으로 가줘"},
            {"role": "assistant", "content": "M:미래관"},
            {"role": "user", "content": "속도 줄여줘"},
            {"role": "assistant", "content": "SL"},
            {"role": "user", "content": predicted_text},
        ]
        filtered_task = self.process_user_input(messages)
        return filtered_task


if __name__ == "__main__":
    stt_agent = STT_Agent()
    stt_agent.setup_mic()

    if stt_agent.listen_for_trigger():
        task = stt_agent.listen_for_task()
        if task:
            logging.info(f"Task recognized: {task}")
        else:
            logging.info("No task was recognized after trigger.")
    else:
        logging.info("No trigger word detected.")