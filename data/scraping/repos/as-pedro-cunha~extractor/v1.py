from extractor.config import openai
from pydantic import BaseModel


class UserDetail(BaseModel):
    name: str
    age: int


user: UserDetail = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)

assert user.name == "Jason"
assert user.age == 25
