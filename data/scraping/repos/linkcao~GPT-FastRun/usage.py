import os

import openai
import yaml

curPath = os.path.dirname(os.path.realpath(__file__))
yamlPath = os.path.join(curPath, "config.yaml")

def read_config():
    f = open(yamlPath, 'r', encoding='utf-8')
    cfg = f.read()
    config_info = yaml.load(cfg)  # 用load方法转字典
    return config_info

cfg = read_config()
os.environ["https_proxy"] = cfg['proxy']

def check():
    success_key = []
    for api_key in cfg['api_keys']:
        try:
            openai.Completion.create(model=cfg['engine'], prompt="Say this is a test", temperature=0, max_tokens=7, api_key = api_key)
            success_key.append(api_key)
            print(api_key)
        except openai.error.RateLimitError as e:
            pass
        except Exception as e:
            pass
    cfg['checked_keys'] = success_key
    print('success check')
    # 写入 yaml 文件
    with open(yamlPath, "w") as yaml_file:
        yaml.dump(cfg, yaml_file)

if __name__ == '__main__':
    check()