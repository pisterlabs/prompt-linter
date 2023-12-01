## whisper 사용하여 대본 추출
import re
from datetime import datetime, timedelta
from openai import OpenAI
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import srt
import deepl
import os  
from dotenv import load_dotenv

def make_mp4s(file_location,mp4_id,langauge):

    scripts_folder = "Script"

    if not os.path.exists(scripts_folder):
        os.makedirs(scripts_folder)
    
    # mp4 폴더가 없으면 생성
    load_dotenv()

# 환경 변수에서 auth_key를 가져와
    auth_key = os.getenv('AUTH_KEY')
    translator = deepl.Translator(auth_key)
    client = OpenAI()

    target_mp4=file_location

    #################### open ai 에서 whisper사용, 영상에서 대본 추출 저장

    if langauge=="en":
        script_id_en=f"{mp4_id}_en"
        script_id_kr=f"{mp4_id}_kr"

        video_clip = VideoFileClip(target_mp4)
        audio_clip = video_clip.audio

        # 음성 저장 폴더 경로
        output_folder = 'audio'

        # 저장 폴더가 없으면 생성
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 추출된 음성 저장
        output_path = os.path.join(output_folder, f'{mp4_id}.mp3')
        audio_clip.write_audiofile(output_path)

        #번역 대본추출
        audio_file= open(output_path, "rb")
        transcript = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file, 
        response_format="srt",
        )

        # 추출된 한국 대본을 저장
        kr_script=transcript.split('\n')
        for i in range(2,len(kr_script),4):
            kr_script[i] = str(translator.translate_text(kr_script[i], target_lang="ko"))
            
        pre_kr_script="\n".join(kr_script)

        with open(os.path.join(scripts_folder, script_id_en+".srt"), "w", encoding="utf-8") as file:
            file.write(transcript)

        with open(os.path.join(scripts_folder, script_id_kr+".srt"), "w", encoding="utf-8") as file:
            file.write(pre_kr_script)
        print("대본 생성완료")

    
    script_id=f"{mp4_id}_{langauge}"

    make_script_mp4(script_id,target_mp4)

    return script_id+".mp4"

    ###################


################### 자막 전처리

# SRT 시간 포맷을 파싱하여 timedelta 객체로 변환 
def parse_srt_time(time_str):
    time_parts = list(map(int, re.split(r"[,:]", time_str)))
    return timedelta(hours=time_parts[0], minutes=time_parts[1], seconds=time_parts[2], milliseconds=time_parts[3])


#자막 형식을 파싱하여 transcriptions에 저장
def make_script_mp4(script_id,target_mp4):
    mp4s_folder = "post_mp4"
    if not os.path.exists(mp4s_folder):
        os.makedirs(mp4s_folder)

    print(script_id+"영상 만들기")
    transcriptions = []
    with open("./Script/"+script_id+".srt", "r",encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines)-1, 4):
            start, end = lines[i+1].strip().split(" --> ")
            text = lines[i+2].strip()
            #출력 자막이 길어지면 두줄로 처리
            parse_len=60
            if "kr" in script_id:
                parse_len =30

            if len(text)>parse_len:
                ## 띄어쓰기로 파싱
                splited_text = text.split(' ')
                sum_n=0
                text=""
                for in_text in splited_text:
                    sum_n+=len(in_text)
                    if sum_n>parse_len:
                        text+=in_text+"\n"
                        sum_n=0
                    else:
                        text+=in_text+" "
            
            start_time = parse_srt_time(start)
            end_time = parse_srt_time(end)

            # 시작 시간과 종료 시간을 초 단위로 변환
            start_seconds = start_time.total_seconds()
            end_seconds = end_time.total_seconds()

            transcriptions.append((start_seconds, end_seconds, text))


    # SRT 객체 생성
    subtitles = [
        srt.Subtitle(index=i, content=text, start=timedelta(seconds=start), end=timedelta(seconds=end))
        for i, (start, end, text) in enumerate(transcriptions)
    ]


    ###################


    ################### MoviePy를 사용하여 영상과 전처리된 자막 합치기

    
    # MoviePy를 사용하여 동영상에 자막 추가
    video = VideoFileClip(target_mp4)

    # 영상에 추가할 자막 폰트 설정
    font_path = "kbo.ttf"
    text_position = ('center', video.size[1]*(0.70))
    # 각 자막에 대한 TextClip 생성 및 위치 설정7
    clips = [video]
    for subtitle in subtitles:

        text_clip = TextClip(subtitle.content, fontsize=36, color='white', font=font_path,bg_color="black")
        # text_clip = text_clip.set_start(subtitle.start.total_seconds()).set_end(subtitle.end.total_seconds()).set_position('bottom')
        text_clip = text_clip.set_start(subtitle.start.total_seconds()).set_end(subtitle.end.total_seconds()).set_position(text_position)
        clips.append(text_clip)

    # 동영상과 자막을 합성
    final_video = CompositeVideoClip(clips)

    # 결과 비디오 파일 저장
    # final_video.write_videofile(script_id+".mp4")
    final_video.write_videofile(os.path.join(mp4s_folder, script_id+".mp4"))
    ###################