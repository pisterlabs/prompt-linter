import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="class Log:\n    def __init__(self, path):\n        dirname = os.path.dirname(path)\n        os.makedirs(dirname, exist_ok=True)\n        f = open(path, \"a+\")\n\n        # Check that the file is newline-terminated\n        size = os.path.getsize(path)\n        if size > 0:\n            f.seek(size - 1)\n            end = f.read(1)\n            if end != \"\\n\":\n                f.write(\"\\n\")\n        self.f = f\n        self.path = path\n\n    def log(self, event):\n        event[\"_event_id\"] = str(uuid.uuid4())\n        json.dump(event, self.f)\n        self.f.write(\"\\n\")\n\n    def state(self):\n        state = {\"complete\": set(), \"last\": None}\n        for line in open(self.path):\n            event = json.loads(line)\n            if event[\"type\"] == \"submit\" and event[\"success\"]:\n                state[\"complete\"].add(event[\"id\"])\n                state[\"last\"] = event\n        return state\n\n\"\"\"\nHere's what the above class is doing, explained in a concise way:\n1.",
  temperature=0,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\"\"\""]
)

# class Log:
#     def __init__(self, path):
#         dirname = os.path.dirname(path)
#         os.makedirs(dirname, exist_ok=True)
#         f = open(path, "a+")

#         # Check that the file is newline-terminated
#         size = os.path.getsize(path)
#         if size > 0:
#             f.seek(size - 1)
#             end = f.read(1)
#             if end != "\n":
#                 f.write("\n")
#         self.f = f
#         self.path = path

#     def log(self, event):
#         event["_event_id"] = str(uuid.uuid4())
#         json.dump(event, self.f)
#         self.f.write("\n")

#     def state(self):
#         state = {"complete": set(), "last": None}
#         for line in open(self.path):
#             event = json.loads(line)
#             if event["type"] == "submit" and event["success"]:
#                 state["complete"].add(event["id"])
#                 state["last"] = event
#         return state

# """
# Here's what the above class is doing, explained in a concise way:
# 1.