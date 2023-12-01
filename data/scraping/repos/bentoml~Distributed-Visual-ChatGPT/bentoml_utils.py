from __future__ import annotations

import inspect
import typing as t
from numpy import str_

import torch
import gradio as gr
import visual_chatgpt as vc
from langchain.llms.openai import OpenAI
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory

import bentoml
from bentoml.io import JSON


# In local mode, ChatBot pass images to models using image's path. In
# distributed mode, ChatBot needs to send the content of image files
# over network to models/runners

def path_to_tuple(path: str):
    with open(path, "rb") as f:
        bs = f.read()
        return (path, bs)


def tuple_to_path(t: tuple[str, bytes]):
    path, bs = t
    with open(path, "wb") as f:
        f.write(bs)
    return path


def path_and_text_to_tuple(path_and_text: str):
    path, _, text = path_and_text.partition(",")
    img_tuple = path_to_tuple(path)
    return img_tuple + (text, )


def tuple_to_path_and_text(t: tuple[str, bytes, str]):
    path, bs, text = t
    path = tuple_to_path((path, bs))
    return ",".join([path, text])


TOOL_DIST_PROCESSORS = {
    # image input, text out
    "ImageCaptioning.inference": {
        "runner_out": lambda captions: captions,
        "api_in": lambda captions: captions,
    },

    # text input, image out
    "Text2Image.inference": {
        "api_out": lambda text: text,
        "runner_in": lambda text: text,
    },

    # image and text input, image out
    "InstructPix2Pix.inference": {
        "api_out": path_and_text_to_tuple,
        "runner_in": tuple_to_path_and_text,
    },
    "PoseText2Image.inference": {
        "api_out": path_and_text_to_tuple,
        "runner_in": tuple_to_path_and_text,
    },
    "SegText2Image.inference": {
        "api_out": path_and_text_to_tuple,
        "runner_in": tuple_to_path_and_text,
    },
    "DepthText2Image.inference": {
        "api_out": path_and_text_to_tuple,
        "runner_in": tuple_to_path_and_text,
    },
    "NormalText2Image.inference": {
        "api_out": path_and_text_to_tuple,
        "runner_in": tuple_to_path_and_text,
    },
    "Text2Box.inference": {
        "api_out": path_and_text_to_tuple,
        "runner_in": tuple_to_path_and_text,
    },

    # image and text input, text out
    "VisualQuestionAnswering.inference": {
        "api_out": path_and_text_to_tuple,
        "runner_in": tuple_to_path_and_text,
        "runner_out": lambda text: text,
        "api_in": lambda text: text,
    },
}


class BaseToolRunnable(bentoml.Runnable):
    pass


# a class to wrap a runner and proxy/adapt model calls to runner calls
class BaseToolProxy:
    TOOL_NAME: str
    RUNNABLE_CLASS: type[BaseToolRunnable]


def make_tool_runnable_method(
    method_name: str,
    processors: dict[str, t.Callable[[t.Any], t.Any]] | None = None,
) -> t.Callable[[BaseToolRunnable, t.Any], t.Any]:

    if processors is None:

        def _run(self: BaseToolRunnable, inputs: t.Any):
            method = getattr(self.model, method_name)
            return method(inputs)

        return _run

    preprocessor = processors.get("runner_in", tuple_to_path)
    postprocessor = processors.get("runner_out", path_to_tuple)

    def _run(self: BaseToolRunnable, inputs: t.Any) -> t.Any:
        method = getattr(self.model, method_name)
        processed_inputs = preprocessor(inputs)
        output = method(processed_inputs)
        processed_output = postprocessor(output)
        return processed_output

    return _run


def make_tool_proxy_method(
    method_name: str,
    processors: dict[str, t.Callable[[t.Any], t.Any]] | None = None,
) -> t.Callable[[BaseToolRunnable, t.Any], t.Any]:

    if processors is None:

        def _run(self: BaseToolProxy, inputs: t.Any):
            runner_method = getattr(self.runner, method_name)
            return runner_method.run(inputs)

        return _run

    # the order is revert for api
    preprocessor = processors.get("api_out", path_to_tuple)
    postprocessor = processors.get("api_in", tuple_to_path)

    def _run(self: BaseToolProxy, inputs: t.Any) -> t.Any:
        runner_method = getattr(self.runner, method_name)
        processed_inputs = preprocessor(inputs)
        output = runner_method.run(processed_inputs)
        processed_output = postprocessor(output)
        return processed_output

    return _run


def create_proxy_class(tool_class: type[object], local: bool = False, gpu: bool = False) -> type[BaseToolProxy]:
    class ToolRunnable(BaseToolRunnable):

        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu") if gpu else ("cpu", )
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = tool_class(self.device)

    class ToolProxy(BaseToolProxy):

        TOOL_NAME = tool_class.__name__
        RUNNABLE_CLASS: type[BaseToolRunnable] = ToolRunnable

        def __init__(self, runner_name: str | None = None):
            if not runner_name:
                runner_name = f"{tool_class.__name__}_runner".lower()
            self.runner = bentoml.Runner(self.RUNNABLE_CLASS, name=runner_name)

    # add method to runnable and proxy model method calls to
    # corresponding runner methods
    for e in dir(tool_class):
        if e.startswith("inference"):

            method = getattr(tool_class, e)

            if local:
                processors = None
            else:
                full_name = f"{tool_class.__name__}.{e}"
                processors = TOOL_DIST_PROCESSORS.get(full_name, dict())

            ToolRunnable.add_method(
                make_tool_runnable_method(e, processors=processors),
                name=e,
                batchable=False,
            )

            model_method = make_tool_proxy_method(e, processors=processors)
            model_method.name = method.name
            model_method.description = method.description
            setattr(ToolProxy, e, model_method)

    return ToolProxy


# helper function to convert EnvVar or cli argument string to load_dict
def parse_load_dict(s: str) -> dict[str, str]:
    return {
        e.split('_')[0].strip(): e.split('_')[1].strip()
        for e in s.split(',')
    }


class BentoMLConversationBot(vc.ConversationBot):
    def __init__(self, load_dict: dict[str, str], local: bool = False):
            
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")

        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, resource in load_dict.items():
            gpu = resource.startswith("cuda")
            tool_class = getattr(vc, class_name)
            proxy_class = create_proxy_class(tool_class, local=local, gpu=gpu)
            self.models[proxy_class.TOOL_NAME] = proxy_class()

        # Load Template Foundation Models
        # for class_name, module in vc.__dict__.items():
        #     if getattr(module, 'template_model', False):
        #         template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
        #         loaded_names = set([type(e).TOOL_NAME for e in self.models.values()
        #                             if not e.template_model])
        #         if template_required_names.issubset(loaded_names):
        #             template_class = getattr(vc, class_name)
        #             self.models[class_name] = template_class(
        #                 **{name: self.models[name] for name in template_required_names})

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output"
        )


def create_gradio_blocks(bot):
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        lang = gr.Radio(choices=["Chinese", "English"], value=None, label="Language")
        chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload an image",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="🖼️", file_types=["image"])

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt, lang], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    return demo


def create_bentoml_service(bot, name="bentoml-visual-chatgpt", gradio_blocks=None):
    runners = [model.runner for model in bot.models.values()]
    svc = bentoml.Service(name, runners=runners)


    # Dummy api endpoint
    @svc.api(input=JSON(), output=JSON())
    def echo(d):
        return d

    if gradio_blocks:
        svc.mount_asgi_app(gradio_blocks.app, path="/ui")

    return svc
