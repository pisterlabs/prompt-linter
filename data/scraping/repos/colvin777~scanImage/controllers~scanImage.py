from fastapi import FastAPI, APIRouter, File, UploadFile
from constants.model_templates import vision_template
import base64
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain.chat_models import ChatOpenAI
import binascii

scamImageRoute = APIRouter()

llm = ChatOpenAI(model="gpt-4-vision-preview", temperature=0, max_tokens=1024)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@scamImageRoute.get("/identifyImage/")
async def identifyImage(image: UploadFile = File(...)):
    sys = """
        - 角色：图像分析师
        - 特长：具备高级视觉能力，用于分析和解释视觉数据。专注于图像中的对象识别和分类。
    """

    # 读取文件内容
    content = await image.read()

    # 将内容转换为 base64
    encoded = binascii.b2a_base64(content, newline=False)
    base64_image = encoded.decode('utf-8')
    prompt = [
        AIMessage(content=sys),
        HumanMessage(content=[
            {"type": "text", "text": vision_template},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            },
        ])
    ]
    result = llm.invoke(prompt)

    print(result)
    return result
