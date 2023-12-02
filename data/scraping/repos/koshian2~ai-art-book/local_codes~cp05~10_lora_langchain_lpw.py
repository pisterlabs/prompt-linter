from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch
from settings import MODEL_DIRECTORY, LORA_DIRECTORY
from utils import load_safetensors_lora

def run_lpw_with_langchain_with_lora(theme, idx, is_gpt_4=True):
    model_name = "gpt-4" if is_gpt_4 else "gpt-3.5-turbo"
    chat = ChatOpenAI(temperature=0.7, model_name=model_name, request_timeout=600)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=chat, memory=memory, verbose=True)
    concept = conversation.predict(input=f"{theme}のイラストを描きたいです。場所、時間、景色に映っているもの、人物、情景、雰囲気を詳細に記述してください")
    prompt = "luminedef, luminernd, "
    prompt += conversation.predict(input="上記の内容を踏まえて一本の英語の文章にしてください")
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    device = "cuda"
    pipe = DiffusionPipeline.from_pretrained(
        f"{MODEL_DIRECTORY}/merge_Counterfeit-V3.0_orangemix", torch_dtype=torch.float16,
         custom_pipeline="lpw_stable_diffusion"
    )
    pipe = load_safetensors_lora(pipe, f"{LORA_DIRECTORY}/lumine1-000008.safetensors", alpha=0.3)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator(device).manual_seed(1234)
    images = pipe.text2img(prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=30,
                           width=512, height=960, max_embeddings_multiples=10, num_images_per_prompt=4).images
    
    with open(f"output/06_gpt_lpw_{idx:03}.txt", "w", encoding="utf-8") as fp:
        fp.write(f"{model_name}\n\n{concept}\n\n{prompt}")
    for i, image in enumerate(images):
        image.save(f"output/10_gpt_lpw_{idx:03}_{i:02}.jpg", quality=92)

def main():
    description = "中学生ぐらいの妹キャラの蛍という名前の金髪の女の子がいます。"
    themes = [
        "蛍が制服姿で大人の男性に勉強を教えてもらいながら、うっかり近くでふざけて笑い、彼の胸に顔を埋める",
        "蛍が風に舞うスカートを押さえながら、大人の男性に恥ずかしがりながら、見ないでと言う",
        "大人の男性が蛍を守るために、彼女を抱きしめているときに、彼女が無邪気に彼に微笑む",
        "蛍が大人の男性の前で、制服から私服に着替える際、うっかりドアを開けてしまい、彼に見られる",
        "蛍が大人の男性と一緒に遊園地に行き、怖い乗り物に乗る際、彼に必死にしがみつく",
        "大人の男性が蛍のことを心配して、彼女の家を訪れたとき、彼女が風呂上がりの姿で現れる",
        "蛍が大人の男性の前で、無邪気に水着姿で海やプールで楽しそうに遊ぶ姿の",
        "蛍が大人の男性に恋愛相談をする際、彼に近づいてきて、うっかり彼の唇に触れる",
        "蛍が大人の男性と一緒に料理をする際、彼に手を引っ張られ、彼の胸に顔を埋める",
        "大人の男性が蛍にプレゼントを渡す際、彼女が嬉しそうな顔で彼に抱きつく。"
    ]
    for i, theme in enumerate(themes):
        run_lpw_with_langchain_with_lora(description+theme, i, is_gpt_4=True)
    
if __name__ == "__main__":
    main()