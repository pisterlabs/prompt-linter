from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch
from settings import MODEL_DIRECTORY
import modal
import subprocess

## setting.pyにOpenAIのAPI Keyを記述

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "langchain", "openai"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_lpw_with_langchain(theme, idx, is_gpt_4=True):
    model_name = "gpt-4" if is_gpt_4 else "gpt-3.5-turbo"
    chat = ChatOpenAI(temperature=0.7, model_name=model_name, request_timeout=600)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=chat, memory=memory, verbose=True)
    concept = conversation.predict(input=f"{theme}のイラストを描きたいです。場所、時間、景色に映っているもの、人物、情景、雰囲気を詳細に記述してください")
    prompt = conversation.predict(input="上記の内容を踏まえて一本の英語の文章にしてください")
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    device = "cuda"
    pipe = DiffusionPipeline.from_pretrained(
        f"{CACHE_DIR}/merge_Counterfeit-V3.0_orangemix", torch_dtype=torch.float16,
         custom_pipeline="lpw_stable_diffusion", safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator(device).manual_seed(1234)
    images = pipe.text2img(prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=30,
                           width=512, height=960, max_embeddings_multiples=10, num_images_per_prompt=4).images
    
    with open(f"{CACHE_DIR}/output/06_gpt_lpw_{idx:03}.txt", "w", encoding="utf-8") as fp:
        fp.write(f"{model_name}\n\n{concept}\n\n{prompt}")
    for i, image in enumerate(images):
        image.save(f"{CACHE_DIR}/output/06_gpt_lpw_{idx:03}_{i:02}.jpg", quality=92)

if __name__ == "__main__":
    with stub.run():
        run_lpw_with_langchain.call("中学生ぐらいのメガネのかわいい女の子が浴衣を着ている", 0, is_gpt_4=True)
    subprocess.run(
        f'modal nfs get model-cache-vol output/06_gpt_lpw_* .', shell=True)

