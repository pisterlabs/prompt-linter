import os
from dotenv import load_dotenv

load_dotenv()

import whisper
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.document_loaders.blob_loaders import Blob
import argparse


# Function to transcribe audio
def transcribe_audio(mp4_path):
    # Load Whisper model
    model = whisper.load_model("medium")

    # Transcribe audio
    result = model.transcribe(mp4_path)
    return result["text"]


# whisperの純粋APIを使うと、ファイルのサイズが大きいとエラーになるので、その辺を考慮しているLangchainを使う
def transcribe_audio_api(mp4_file):
    print("transcribe_audio_api")
    # Create a Blob object from the mp4_path
    # blob = Blob.from_path(mp4_path)

    blob = Blob.from_data(data=mp4_file, mime_type="video/mp4")

    full_transcription = " ".join(
        document.page_content for document in OpenAIWhisperParser().lazy_parse(blob)
    )
    return full_transcription


# Function to get GPT-4 completion
def get_completion(texts, model):
    llm = ChatOpenAI(
        temperature=0,
        model=model,
    )
    prompt = PromptTemplate(
        input_variables=["text"],
        template="##依頼以下の内容は、会議の文字起こしです。\r\nこれを、わかりやすく箇条書きで、内容を補完、修正しながら、体系的に記載し直してください。それぞれ各章のわかりやすいタイトルをつけて、説明に入ってください。説明は、MTG内容が漏れないように具体的かつ詳細に記載してください。\r\n出力する前に内容をもう一度確認して、内容に誤りや漏れがあるようなら修正してください。 \r\n##出力形式 \r\n# (title)\r\n## (sub_title)\r\n- (details)\r\n\r\n##内容\r\n{text}\r\n",
    )

    llm = LLMChain(llm=llm, prompt=prompt)
    responses = []
    all_tokens = 0
    # 円換算レート
    rate = 130
    price_per_1ktoken = 0.004 * rate

    for text in texts:
        with get_openai_callback() as callback:
            response = llm.run(text=text)
            responses.append(response)

            # トークン数をカウント
            print("tokens : " + str(callback.total_tokens))
            print("price : " + str((callback.total_tokens / 1000) * price_per_1ktoken))
            # 全てのトークン数積み上げる
            all_tokens += callback.total_tokens

    mergeresponse = "\n\n".join(responses)
    print("total tokens : " + str(all_tokens))
    print("total price : " + str((all_tokens / 1000) * price_per_1ktoken))

    return mergeresponse


# Function to set vectorstore
# If transcription file exists, load transcription from file
def split_file_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()

    # chnk_sizeは、分割する文字数
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # ループの回数をカウントする変数
    count = 0
    for text in texts:
        # ループの回数をカウント
        count += 1
        print(str(count) + "つ目 : " + str(text))

    return texts


# Function to handle transcription and GPT-4 response
def handle_audio(mp4_file, result_path, api, gptmodel):
    # Generate transcription file path
    transcription_file = result_path + ".txt"

    # Check if transcription file exists
    # If it does, read transcription from file
    if os.path.isfile(transcription_file):
        transcription = split_file_text(transcription_file)

    else:
        # 二つ目の引数オプションがapiの場合、apiを利用する
        if api == "api":
            # Transcribe audio and save transcription
            transcription_text = transcribe_audio_api(mp4_file)
        else:
            # Transcribe audio and save transcription
            transcription_text = transcribe_audio(mp4_file)

        with open(transcription_file, "w") as file:
            file.write(transcription_text)

        transcription = split_file_text(transcription_file)

    # Get GPT-4 completion
    response = get_completion(transcription, model=gptmodel)
    print(response)

    # responseを別のファイルに書き込む
    with open(result_path + "response.txt", "w") as file:
        file.write(response)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper and gptmodel options")

    parser.add_argument("mp4_path", type=str, help="path to the mp4 file")
    parser.add_argument(
        "-W",
        "--whisper",
        type=str,
        required=False,
        help="whisper API or install option",
        default="api",
    )
    parser.add_argument(
        "-G",
        "--gpt",
        type=str,
        required=False,
        help="gptmodel option",
        default="gpt-3.5-turbo-16k",
    )
    args = parser.parse_args()

    mp4_file = open(args.mp4_path, "rb").read()

    result_path = os.path.splitext(args.mp4_path)[0]

    # 議事録作成処理の実行
    try:
        handle_audio(
            mp4_file,
            result_path,
            args.whisper,
            args.gpt,
        )
    except Exception as e:
        print(e)
