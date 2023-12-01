# coding=utf-8
import yaml
import pathlib
import openai
from utils import get_config
from utils.openai_wrapper import ChatGPT

base_path = pathlib.Path.cwd().parent
config_file = base_path / 'config' / 'config.yaml'

with open(config_file, 'r') as inf:
    config = yaml.safe_load(inf)
config = get_config.run(config_file)

def main():
    text = """
    以下の文章を魅力的にしてください。
    私は株式会社Aにおいて、webサイトの行動履歴を用いて、どのような行動をしているユーザーがアクションをする割合を算出し、     予測したアクション確率によって支援の内容を変えるという施策を実施してきました。     また、アイテムの推薦システムのアルゴリズムの開発を行い、     recboleやbert4recを用いて推薦するアルゴリズムの検証・システムの構築・効果検証を実践してきました。
    """
    chatgpt = ChatGPT(config.openai_api_key, config.openai_model)
    print(chatgpt.run_chat(text))

if __name__ == '__main__':
    main()
