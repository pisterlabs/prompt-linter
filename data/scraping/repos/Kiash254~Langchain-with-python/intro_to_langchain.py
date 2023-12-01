{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 'null',
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pfgRURlWP-At",
        "outputId": "6920694d-e368-4697-ac48-c92a3541e789"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting langchain\n",
            "  Downloading langchain-0.0.311-py3-none-any.whl (1.8 MB)\n",
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/1.8 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.3/1.8 MB\u001b[0m \u001b[31m7.4 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m1.8/1.8 MB\u001b[0m \u001b[31m27.9 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.8/1.8 MB\u001b[0m \u001b[31m21.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (6.0.1)\n",
            "Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.0.21)\n",
            "Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.10/dist-packages (from langchain) (3.8.5)\n",
            "Requirement already satisfied: anyio<4.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (3.7.1)\n",
            "Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (4.0.3)\n",
            "Collecting dataclasses-json<0.7,>=0.5.7 (from langchain)\n",
            "  Downloading dataclasses_json-0.6.1-py3-none-any.whl (27 kB)\n",
            "Collecting jsonpatch<2.0,>=1.33 (from langchain)\n",
            "  Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)\n",
            "Collecting langsmith<0.1.0,>=0.0.43 (from langchain)\n",
            "  Downloading langsmith-0.0.43-py3-none-any.whl (40 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m40.0/40.0 kB\u001b[0m \u001b[31m4.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: numpy<2,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain) (1.23.5)\n",
            "Requirement already satisfied: pydantic<3,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain) (1.10.13)\n",
            "Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.10/dist-packages (from langchain) (2.31.0)\n",
            "Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from langchain) (8.2.3)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (23.1.0)\n",
            "Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (3.3.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.0.4)\n",
            "Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.9.2)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.4.0)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<4.0->langchain) (3.4)\n",
            "Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<4.0->langchain) (1.3.0)\n",
            "Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<4.0->langchain) (1.1.3)\n",
            "Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain)\n",
            "  Downloading marshmallow-3.20.1-py3-none-any.whl (49 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m49.4/49.4 kB\u001b[0m \u001b[31m5.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7,>=0.5.7->langchain)\n",
            "  Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)\n",
            "Collecting jsonpointer>=1.9 (from jsonpatch<2.0,>=1.33->langchain)\n",
            "  Downloading jsonpointer-2.4-py2.py3-none-any.whl (7.8 kB)\n",
            "Requirement already satisfied: typing-extensions>=4.2.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1->langchain) (4.5.0)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (2.0.6)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2->langchain) (2023.7.22)\n",
            "Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain) (3.0.0)\n",
            "Requirement already satisfied: packaging>=17.0 in /usr/local/lib/python3.10/dist-packages (from marshmallow<4.0.0,>=3.18.0->dataclasses-json<0.7,>=0.5.7->langchain) (23.2)\n",
            "Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain)\n",
            "  Downloading mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)\n",
            "Installing collected packages: mypy-extensions, marshmallow, jsonpointer, typing-inspect, langsmith, jsonpatch, dataclasses-json, langchain\n",
            "Successfully installed dataclasses-json-0.6.1 jsonpatch-1.33 jsonpointer-2.4 langchain-0.0.311 langsmith-0.0.43 marshmallow-3.20.1 mypy-extensions-1.0.0 typing-inspect-0.9.0\n"
          ]
        }
      ],
      "source": [
        "!pip install langchain"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 'null',
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E5LKHvZoQJyM",
        "outputId": "cfb418ad-ea6d-4f86-ff4e-79f1305da788"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting openai\n",
            "  Downloading openai-0.28.1-py3-none-any.whl (76 kB)\n",
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/77.0 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m77.0/77.0 kB\u001b[0m \u001b[31m2.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: requests>=2.20 in /usr/local/lib/python3.10/dist-packages (from openai) (2.31.0)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai) (4.66.1)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from openai) (3.8.5)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (3.3.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2.0.6)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2023.7.22)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (23.1.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (6.0.4)\n",
            "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (4.0.3)\n",
            "Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.9.2)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.4.0)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.3.1)\n",
            "Installing collected packages: openai\n",
            "Successfully installed openai-0.28.1\n"
          ]
        }
      ],
      "source": [
        "pip install openai"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "MT_qo294Qzef"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "os.environ[\"OPENAI_API_KEY\"]='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'"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "z7XD4OHKb9K1"
      },
      "source": [
        "## Building a Language model Application\n",
        "\n",
        "LLMS:Get Preductions from a Langauge Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "u_ZQLJrWbPhg"
      },
      "outputs": [],
      "source": [
        "from langchain.llms import OpenAI"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "zZ3EYv3fcL8n"
      },
      "outputs": [],
      "source": [
        "llm=OpenAI(temperature=0.9)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "arcT_O7scS1M"
      },
      "outputs": [],
      "source": [
        "text=\"Who is the Fifth President of Kenya\"\n",
        "print(llm(text))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "k8I5FY8Tf00x"
      },
      "source": [
        "## promt Template:Manage Prompts For LLMs"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "yTOAlgeicopf"
      },
      "outputs": [],
      "source": [
        "from langchain.prompts import PromptTemplate"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "9SAlKZY1gHM-"
      },
      "outputs": [],
      "source": [
        "prompt=PromptTemplate(\n",
        "    input_variables=[\"food\"],\n",
        "    template=\"What are the first five desinations for someone who likes to eat {food} ?\",\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SCMFCMw4gvNy",
        "outputId": "cc063a91-b94f-4df0-fb4a-9900e4f7ab2e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "What are the first five desinations for someone who likes to eat dessert ?\n"
          ]
        }
      ],
      "source": [
        "print(prompt.format(food=\"dessert\"))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "A35pj7KwhlzS"
      },
      "outputs": [],
      "source": [
        "print(llm(prompt.format(food=\"dessert\")))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "m-Si5BURiFBC"
      },
      "source": [
        "## Langchains :combining LLMs and Prompts in multi-step workflows"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "6wONXtD8h45i"
      },
      "outputs": [],
      "source": [
        "from langchain.prompts import PromptTemplate\n",
        "from langchain.llms import OpenAI\n",
        "from langchain.chains import LLMChain"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "g0mG-V-ri_6B"
      },
      "outputs": [],
      "source": [
        "llm=OpenAI(temperature=0.9)\n",
        "\n",
        "prompt=PromptTemplate(\n",
        "    input_variables=['food'],\n",
        "    template=\"What are 5 Vacation destination for someone who likes to eat {food}\",\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "mpyTP29ckZqR"
      },
      "outputs": [],
      "source": [
        "chain=LLMChain(llm=llm,prompt=prompt)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "wySRbBAdkjFM"
      },
      "outputs": [],
      "source": [
        "print(chain.run('fruit'))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Df-6YaaOk7vw"
      },
      "source": [
        "#Agents:Dynamically Cell Chains based on User Input"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "RgDkO0NYknpS"
      },
      "outputs": [],
      "source": [
        " from langchain.agents import load_tools\n",
        " from langchain.agents import initialize\n",
        " from langchain.llms import OpenAI"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "id": "qUtQ85krlVBo"
      },
      "outputs": [],
      "source": [
        "#Load the model\n",
        "llm=OpenAI(temperature=0)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "id": "GWLHI0r6ldkn"
      },
      "outputs": [],
      "source": [
        "#load  some tools to use\n",
        "os.environ[\"SERPAPI_API_KEY\"]=''\n",
        "\n",
        "tools=load_tools(['serpapi','llm-math'],llm=llm)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "v8_aUmiwl0EC"
      },
      "outputs": [],
      "source": [
        "#here  I will initialize the agents\n",
        "# 1. The tools\n",
        "#2. the Language model\n",
        "#3. The type of the agent we want t o use\n",
        "\n",
        "agent=initialize.initialize_agent(tools,llm,agent=\"zero-shot-react_description\",verbose=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "uPnNNkXpn3fk"
      },
      "outputs": [],
      "source": [
        "agent.run(\"Who us current president of kenya? what is the largest Prime number that is smaller than thier age \")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3YZ5sjf3ofaJ"
      },
      "source": [
        "## Memory: Add State to Chains and Agents"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "52JwIyILoT_n"
      },
      "outputs": [],
      "source": [
        "from langchain import OpenAi, ConversationChain"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "s_KhUwmxoZlt"
      },
      "outputs": [],
      "source": [
        "llm=OpenAI=(temparature=0)\n",
        "conversion=ConversationChain(llm=llm,verbose=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "conversion.predict(input=\"Hi There\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "conversion.predict(input=\"Ai am Doing Well What about You Please\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "conversion.predict(\"I wish to sleep in on room with you will you mind\")"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
