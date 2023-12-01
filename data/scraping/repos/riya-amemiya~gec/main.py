import guidance
from dotenv import load_dotenv
import os
import json

load_dotenv()
model = ""


def main():
    def gptConfig():
        global model
        with open("gpt.config.json") as f:
            config = json.loads(f.read())

        gptConfig = ""

        for key in config:
            if key == "model":
                model = config[key]
                continue
            else:
                gptConfig += f" {key}={config[key]}"

        return f"{{{{gen 'res'{gptConfig}}}}}"

    # set the default language model used to execute guidance programs

    ask = None
    tmpAsk = None
    res = None
    output = ""
    gptOption = gptConfig()
    guidance.llm = guidance.llms.OpenAI(model=model)

    def parse(data):
        if data.get("flag") is None:
            data["flag"] = "true"
        elif data["flag"] == "true":
            if data["context"] == "{{ask}}" and ask is not None:
                data["context"] = ask
            elif data["context"] == gptOption and res is not None:
                data["context"] = res
            return data
        if data.get("context") is None:
            if data["role"] == "assistant":
                data["context"] = gptOption
            elif data["role"] == "user":
                data["context"] = "{{ask}}"
        data["role"] = ["{{#" + data["role"] + "~}}", "{{~/" + data["role"] + "}}"]

        return data

    def create_pronpt(data):
        return f"""{data["role"][0]}
    {data["context"]}
    {data["role"][1]}
    """

    data = [
        {
            "role": "user",
        },
        {
            "role": "assistant",
        },
    ]

    tmp = list(map(parse, data))
    pronpt = "".join(list(map(create_pronpt, tmp)))
    try:
        while True:
            if tmpAsk is None:
                ask = input(">>")
            if ask == "exit":
                break
            elif ask == "save":
                tmpData = data
                tmpData = list(map(parse, tmpData))
                tmpPronpt = pronpt
                tmpPronpt = "".join(list(map(create_pronpt, tmpData)))
                if os.path.exists("save") is False:
                    os.mkdir("save")
                id = input("save id:")
                if os.path.exists(f"save/{id}") is False:
                    os.mkdir(f"save/{id}")
                else:
                    print("already exists")
                    overRide = input("override? [y/n]:")
                    if overRide == "y":
                        pass
                    else:
                        continue
                with open(f"save/{id}/log.txt", mode="w") as f:
                    f.write(output)
                with open(f"save/{id}/pronpt.txt", mode="w") as f:
                    f.write(str(tmpPronpt))
                with open(f"save/{id}/data.txt", mode="w") as f:
                    f.write(json.dumps({"data": tmpData}))
                continue
            elif ask == "read":
                with open("ask.txt", mode="r") as f:
                    tmpAsk = f.read()
                    ask = None
                continue
            elif ask == "load":
                id = input("load id:")
                with open(f"save/{id}/log.txt", mode="r") as f:
                    output = f.read()
                with open(f"save/{id}/data.txt", mode="r") as f:
                    data = json.loads(f.read())["data"]
                continue
            elif ask == "nowData":
                print(data)
                print(pronpt)
                continue
            elif ask == "reset":
                data = [
                    {
                        "role": "user",
                    },
                    {
                        "role": "assistant",
                    },
                ]
                tmp = list(map(parse, data))
                pronpt = "".join(list(map(create_pronpt, tmp)))
                ask = None
                res = None
                with open("log.txt", mode="w") as f:
                    f.write("")
                continue
            elif ask == "load_config":
                gptOption = gptConfig()
                guidance.llm = guidance.llms.OpenAI(model=model)
                continue
            elif ask == "show_config":
                print(model)
                print(gptOption)
                continue
            tmp = list(map(parse, data))
            pronpt = "".join(list(map(create_pronpt, tmp)))
            program = guidance(pronpt)
            if tmpAsk:
                ask = tmpAsk
                tmpAsk = None
            executed_program = program(
                ask=ask,
            )
            ask = executed_program["ask"]
            res = executed_program["res"]
            data += [
                {
                    "role": "user",
                },
                {
                    "role": "assistant",
                },
            ]
            with open("log.txt", mode="w") as f:
                if not isinstance(res, str) and len(res) > 1:
                    res = "".join(list(map(lambda x: x + "\n\n", res)))
                print(res)
                output += "\n" + f"user:{ask}" + "\n" + f"ai:{res}" + "\n"
                f.write(output)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
