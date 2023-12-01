import os
import glob
import json
import time
import openai

from query_openai import queryQuestions
from text_message import TextMessage, parseKISASpamDataFile







if __name__ == "__main__" :
    # openAI API key를 이 파일에 저장하세요.
    # [KEY1, KEY2, ... ] 와 같은 형식으로 저장하세요.
    # 여기서는 1개의 키만 사용합니다.
    OPENAI_API_KEY_FILE_PATH = "./openai_api_keys.json"

    # chatGPT 모델 이름을 선택하세요
    # https://platform.openai.com/docs/models/overview
    GPT_MODEL_NAME = "gpt-3.5-turbo"

    # KISA 에서 전달받은 스팸 문자 csv 파일을 이 디렉토리에 저장하세요
    DATA_DIR_PATH = "./data/"
    assert os.path.exists(DATA_DIR_PATH), f"{DATA_DIR_PATH} 디렉토리가 존재하지 않습니다."

    # 결과를 저장할 디렉토리를 지정하세요
    SAVE_FILE_DIR_PATH = "./results/"
    assert os.path.exists(SAVE_FILE_DIR_PATH), f"{SAVE_FILE_DIR_PATH} 디렉토리가 존재하지 않습니다."

    # chatGPT에 보낼 쿼리의 질문을 지정하세요
    QUERY_QUESTION = "다음 문자는 스팸 문자일까, 아닐까? \n"

    # read openAI API key
    with open(OPENAI_API_KEY_FILE_PATH, "r") as fp :
        openai_api_key = json.load(fp)[0]
        # openai API 키 인증
        openai.api_key = openai_api_key

    # read KISA-provided spam data file path 
    spam_data_file_path_list = glob.glob(os.path.join(DATA_DIR_PATH, "*.csv"))
    print("KISA 스팸 문자 데이터 파일 목록 :")
    for idx, file_path in enumerate(spam_data_file_path_list) :
        print(f"{idx}\t {file_path}")
    print()

    spam_data_file_idx = input("처리할 스팸 문자 데이터 파일의 인덱스를 입력하세요 : ")
    print()
    SPAM_DATA_PATH = spam_data_file_path_list[int(spam_data_file_idx)]

    print("처리할 파일 :")
    print(SPAM_DATA_PATH)

    spam_data_list = parseKISASpamDataFile(SPAM_DATA_PATH)
    for data in spam_data_list :
        data.query_content = QUERY_QUESTION + data.content


    FINISH_FLAG = False
    while not FINISH_FLAG :
        queryQuestions(
            model_name = GPT_MODEL_NAME,
            spam_data_list = spam_data_list[:20]
        )

        KEY = input("남은 쿼리를 이어서 수행하려면 아무 키나 누르세요. 종료하려면 q를 누르세요 : ")
        if KEY in ["q", "Q"] :
            FINISH_FLAG = True

    RESULT_FILE_NAME = "{}_processed_{}.json".format(
        os.path.basename(SPAM_DATA_PATH).split(".")[0],
    time.strftime("%Y%m%d%H%M%S")
    )
    RESULT_FILE_PATH = os.path.join(SAVE_FILE_DIR_PATH, RESULT_FILE_NAME)

    print("saving result to {}".format(RESULT_FILE_PATH))
    with open(RESULT_FILE_PATH, "w") as fp :
        json.dump(
            list(map(
                lambda data : data.toDict(),
                spam_data_list
            )),
            fp,
            indent=2,
            ensure_ascii=False
        )
    print("file saved")