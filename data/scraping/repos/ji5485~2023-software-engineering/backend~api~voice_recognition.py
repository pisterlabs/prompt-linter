import json
from openai import OpenAI
from starlette.config import Config
from packages.Spot import Hazard, ColorBlob
from packages.Position import Position

config = Config(".env")

functions = [
  {
    "name": "create_hazard_spot",
    "description": "x, y 좌표를 전달하면 위험 지점을 생성한다.",
    "parameters": {
      "type": "object",
      "properties": {
        "x": {"type": "number", "description": "x 좌표를 입력받는다."},
        "y": {"type": "number", "description": "y 좌표를 입력받는다."},
      },
      "required": ["x", "y"],
    }
  },
  {
    "name": "create_color_blob_spot",
    "description": "x, y 좌표를 전달하면 중요 지점을 생성한다.",
    "parameters": {
      "type": "object",
      "properties": {
        "x": {"type": "number", "description": "x 좌표를 입력받는다."},
        "y": {"type": "number", "description": "y 좌표를 입력받는다."},
      },
      "required": ["x", "y"],
    }
  }
]

class VoiceRecognition:
  def __init__(self, add_on):
    self.add_on = add_on
    self.client = OpenAI(api_key=config("OPENAI_API_KEY"))
    self.registered_functions = {
      "create_hazard_spot": self.create_hazard_spot,
      "create_color_blob_spot": self.create_color_blob_spot,
    }
  
  def recognize(self, audio):
    transcript = self.client.audio.transcriptions.create(model="whisper-1", file=audio)
    return transcript.text
  
  def parse(self, text):
    response = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{ "role": "user", "content": text }],
      functions=functions,
      function_call="auto"
    )

    if response.choices[0].message.function_call:
      name = response.choices[0].message.function_call.name
      args = json.loads(response.choices[0].message.function_call.arguments)
      result = self.registered_functions[name](args.get("x"), args.get("y"))

      return { "status": True, "spot": result }
    else:
      return { "status": False, "spot": None }

  def create_hazard_spot(self, x, y):
    self.add_on.create_spot(Hazard(), Position(x, y))
    return { "spot": "hazard", "position": { "x": x, "y": y } }

  def create_color_blob_spot(self, x, y):
    self.add_on.create_spot(ColorBlob(), Position(x, y))
    return { "spot": "color_blob", "position": { "x": x, "y": y } }

