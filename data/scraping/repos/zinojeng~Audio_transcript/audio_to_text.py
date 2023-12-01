{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "private_outputs": true,
      "provenance": [],
      "mount_file_id": "1tMPz1SVIpQzR_W-htdOg7yfLcL08yWAl",
      "authorship_tag": "ABX9TyOCT3vDlUo/MZLek9JS6MWy",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/zinojeng/Audio_transcript/blob/main/audio_to_text.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install openai langchain tiktoken"
      ],
      "metadata": {
        "id": "XSwxNiY3OmCn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "EFtMNeAyGeE3"
      }
    },
    {
      "cell_type": "code",
      "source": [
        " from langchain.chat_models import ChatOpenAI"
      ],
      "metadata": {
        "id": "tiZMDIPwQaCt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter, TokenTextSplitter"
      ],
      "metadata": {
        "id": "ghORR9FdSvxE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain.docstore.document import Document"
      ],
      "metadata": {
        "id": "hS6oGk_XUtEe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain.prompts import PromptTemplate"
      ],
      "metadata": {
        "id": "eUwrnuuQWXpY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain.chains.summarize import load_summarize_chain"
      ],
      "metadata": {
        "id": "bpwiyUsrWq5E"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jYWyZ4zfL9BM"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Set your media file path. ex: /content/drive/MyDrive/Colab Notebooks/Files/xxxx.m4a\n",
        "media_file_path = \"/content/drive/MyDrive/Colab Notebooks/Files/Metformin Pregnancy/Rebatle NO.m4a\"\n",
        "# Open the media file\n",
        "media_file = open(media_file_path, \"rb\")"
      ],
      "metadata": {
        "id": "i2oTwrx3M5DY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Set your model ID\n",
        "model_id = \"whisper-1\""
      ],
      "metadata": {
        "id": "uDFBQRUtOM86"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize LLM\n",
        "import openai\n",
        "api_key=\"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\"\n",
        "llm = ChatOpenAI(\n",
        "    openai_api_key = api_key,\n",
        "    temperature=0.2,\n",
        "    model_name=\"gpt-3.5-turbo-16k\"\n",
        "    )"
      ],
      "metadata": {
        "id": "uYVVebH0Pfvi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Call the API\n",
        "response = openai.Audio.transcribe(\n",
        "    api_key=api_key,\n",
        "    model=model_id,\n",
        "    file=media_file\n",
        ")"
      ],
      "metadata": {
        "id": "zUcJLY8SOPXr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Assignthetranscripttoavariable\n",
        "transcript = response[\"text\"]"
      ],
      "metadata": {
        "id": "VJVmD7ipSS8t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Splitthetext\n",
        "text_splitter = TokenTextSplitter(model_name=\"gpt-3.5-turbo-16k\", chunk_size=8000,\n",
        "chunk_overlap=300)\n",
        "texts = text_splitter.split_text(transcript)"
      ],
      "metadata": {
        "id": "QINbsfZASWRs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Create documents for further processing\n",
        "docs = [Document(page_content=t) for t in texts]"
      ],
      "metadata": {
        "id": "QRj0ebotToaf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prompt_template_summary = \"\"\"\n",
        "You are a management assistant with a specialization in diabetes. You are takingnotes for a meeting.\n",
        "Write a detailed summary of the following transcript of a meeting:\n",
        "\n",
        "{text}\n",
        "\n",
        "Make sure you don't lose any important information. Be as detailed as possible in your\n",
        "summary and list 10 most important keypoints in zh-tw\n",
        "Also end with a list of:\n",
        "- Main takeaways\n",
        "- Action items\n",
        "- Decisions\n",
        "- Open questions\n",
        "- Next steps\n",
        "If there are any follow-up meetings, make sure to include them in the summary and\n",
        "mentioned it specifically.\n",
        "\n",
        "DETAILED SUMMARY IN zh-tw:\"\"\"\n",
        "PROMPT_SUMMARY = PromptTemplate(template=prompt_template_summary, input_variables=[\"text\"])\n",
        "\n",
        "refine_template_summary = (\n",
        "'''\n",
        "You are a management assistant with a specialization in note taking. You are taking\n",
        "notes for a meeting.\n",
        "Your job is to provide detailed summary of the following transcript of a meeting:\n",
        "We have provided an existing summary up to a certain point: {existing_answer}.\n",
        "We have the opportunity to refine the existing summary (only if needed) with some more context below.\n",
        "----------\n",
        "{text}\n",
        "----------\n",
        "Given the new context, refine the original summary in zh-tw. If the context isn't useful, return the original summary. Make sure you are detailed in\n",
        "your summary.\n",
        "Make sure you don't lose any important information. Be as detailed as possible.\n",
        "Summary 10 keypoints\n",
        "\n",
        "Also end with a list of:\n",
        "\n",
        "- Main takeaways\n",
        "- Action items\n",
        "- Decisions\n",
        "- Open questions\n",
        "- Next steps\n",
        "\n",
        "If there are any follow-up meetings, make sure to include them in the summary and\n",
        "mentioned it specifically.\n",
        "\n",
        "'''\n",
        ")\n",
        "refine_prompt_summary = PromptTemplate(\n",
        "    input_variables=[\"existing_answer\", \"text\"],\n",
        "    template=refine_template_summary,\n",
        ")\n"
      ],
      "metadata": {
        "id": "sB2L0vTdWNkd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Create a summary\n",
        "sum_chain = load_summarize_chain(llm, chain_type=\"refine\", verbose=True,\n",
        "                                question_prompt=PROMPT_SUMMARY,\n",
        "                                refine_prompt=refine_prompt_summary)"
      ],
      "metadata": {
        "id": "zkeCbJffWg-b"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "summary = sum_chain.run(docs)"
      ],
      "metadata": {
        "id": "uE3BSVpGWwRn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Export the summary to a text file\n",
        "with open(\"summary.txt\", \"w\") as f:\n",
        "    f.write(summary)"
      ],
      "metadata": {
        "id": "bsEyQCejW4ly"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(transcript)"
      ],
      "metadata": {
        "id": "G0yubX_wXX4f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(summary)"
      ],
      "metadata": {
        "id": "4q1LqF-VW-8l"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#translate to Chinese"
      ],
      "metadata": {
        "id": "PR-bTJPXB15d"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain.chat_models import ChatOpenAI\n",
        "from langchain import LLMChain\n",
        "\n",
        "from langchain.prompts.chat import (\n",
        "    ChatPromptTemplate,\n",
        "    SystemMessagePromptTemplate,\n",
        "    HumanMessagePromptTemplate,\n",
        ")\n",
        "\n",
        "template = \"\"\"You are a endocrinologist and diabetes specialist and helpful assistant that translates Sentence by sentence, first by  {input_language} and then lines break and show {output_language} sentence by sentence  \"\"\"\n",
        "\n",
        "system_message_prompt = SystemMessagePromptTemplate.from_template(template)\n",
        "\n",
        "human_template = \"{text}\"\n",
        "human_message_prompt = HumanMessagePromptTemplate.from_template(human_template) #易忽略\n",
        "translation_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])\n",
        "\n",
        "chat = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name='gpt-3.5-turbo-16k')\n",
        "#result = chat(translation_prompt.format_messages(input_language=\"English\", output_language=\"zh-tw\", text=transcript))\n",
        "\n",
        "chain = LLMChain(llm=chat, prompt=translation_prompt)\n",
        "chain.run(input_language=\"English\", output_language=\"zh-tw\", text=transcript)\n",
        "\n",
        "\n",
        "#translated_text = response[\"text\"]\n",
        "#print(result)\n"
      ],
      "metadata": {
        "id": "vsPZDqLF_ZB6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(docs)"
      ],
      "metadata": {
        "id": "I5h6vrAN_D26"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}