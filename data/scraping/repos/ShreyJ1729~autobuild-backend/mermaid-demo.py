import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import modal
import openai
import dotenv

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

stub = modal.Stub("autobuild", image=image)


def load_openai_key():
    dotenv.load_dotenv("./.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")


@app.get("/")
async def root():
    return {"message": "AutoBuild Backend"}


def build_message_list(variables, prompt_path, prompt_instructions):
    prompt = open(prompt_path, "r").read()

    # replace variables in prompt
    for key in variables:
        prompt = prompt.replace(f"{{{{{key}}}}}", variables[key])

    # build and return messageList
    return [
        {
            "role": "system",
            "content": prompt_instructions,
        },
        {
            "role": "assistant",
            "content": "Got it!",
        },
        {"role": "user", "content": prompt},
    ]


class MermaidGenRequest(BaseModel):
    description: str


@app.post("/mermaid-gen")
async def mermaid_gen(data: MermaidGenRequest, response: Response):
    print("mermaid-gen request received: ", data.description)

    load_openai_key()
    messageList = build_message_list(
        variables={
            "DESCRIPTION": data.description,
        },
        prompt_path="./prompts/mermaid-demo/mermaid_gen.txt",
        prompt_instructions="You are a helpful markdown generation bot for mermaid diagrams that architects mermaid diagrams for React web apps in markdown from a text description. Stop token: <<|END|>>",
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messageList
    )

    mermaid = completion.choices[0].message.content.rstrip("<<|END|>>")

    print(mermaid)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return {"mermaid": mermaid}


class MermaidEditRequest(BaseModel):
    mermaid: str
    query: str


@app.post("/mermaid-edit")
async def mermaid_edit(data: MermaidEditRequest):
    print("mermaid-edit request received: ", data.mermaid, "\n", data.query)

    load_openai_key()
    messageList = build_message_list(
        variables={
            "MERMAID": data.mermaid,
            "QUERY": data.query,
        },
        prompt_path="./prompts/mermaid-demo/mermaid_edit.txt",
        prompt_instructions="You are a helpful markdown generation bot for mermaid diagrams that takes in a markdown mermaid diagram and a query and returns a new mermaid diagram with edits. Stop token: <<|END|>>",
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messageList
    )

    mermaid = completion.choices[0].message.content.rstrip("<<|END|>>")

    print(mermaid)
    return {"mermaid": mermaid}


def component_list_gen(mermaid: str):
    print("component-list-gen request received: ", mermaid)

    load_openai_key()
    messageList = build_message_list(
        variables={
            "MERMAID": mermaid,
        },
        prompt_path="./prompts/mermaid-demo/component_list_gen.txt",
        prompt_instructions="You are a helpful markdown parsing bot that takes in a markdown mermaid diagram and returns a list of components in a bottom-up traversal. Stop token: <<|END|>>",
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messageList
    )

    component_list = (
        completion.choices[0]
        .message.content.rstrip("<<|END|>>")
        .rstrip("\n")
        .rstrip(" ")
    )

    print(component_list)
    return {"component_list": component_list}


def build_prompt(variables, prompt_path):
    prompt = open(prompt_path, "r").read()

    # replace variables in prompt
    for key in variables:
        prompt = prompt.replace(f"{{{{{key}}}}}", variables[key])

    return prompt


class MermaidToCodeRequest(BaseModel):
    mermaid: str
    description: str


@app.post("/mermaid-to-code")
async def mermaid_to_code(data: MermaidToCodeRequest):
    # print("mermaid-to-code request received: ", data.description, data.mermaid)
    load_openai_key()

    traversalList = component_list_gen(data.mermaid)
    traversalList = traversalList["component_list"]
    traversalList = traversalList.lstrip("[").rstrip("]")

    # print("traversalList String: <<", traversalList, ">>")
    # convert string of list into list
    traversalList = traversalList.split(",")

    # print("traversalList: ", traversalList)

    all_code = {}

    for component in traversalList:
        print(component)
        messageList = build_message_list(
            variables={
                "MERMAID": data.mermaid,
                "FILENAME": component,
            },
            prompt_path="./prompts/mermaid-demo/mermaid_to_code.txt",
            prompt_instructions="You are a helpful Typescript React code generation bot that takes in a Typescript React App description, a filename and markdown mermaid diagram architecting the React app and you return the code for that file and ONLY that file. You do not import from any file or module that is not specified in the user-provided mermaid diagram. You import children component of a file that are shown in the markdown mermaid diagram. You always define a component's prop types in the same file as the component using the PropTypes module. You use tailwind css and create stunning, modern and sleek UI designs. Stop token: <<|END|>>",
        )

        print(messageList)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messageList
        )

        tsx_code = completion.choices[0].message.content.rstrip("<<|END|>>")
        all_code[component] = tsx_code

        print("--------------------")
        print(component)
        print("```tsx")
        print(tsx_code)
        print("```")

    return {"all_code": all_code}


@stub.asgi(
    mounts=[
        modal.Mount.from_local_dir("./prompts", remote_path="/root/prompts"),
        modal.Mount.from_local_file("./.env", remote_path="/root/.env"),
    ]
)
def fastapi_app():
    return app
