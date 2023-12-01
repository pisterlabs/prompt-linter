#    Jobzilla automates job search using a LLM.
#    Copyright (C) 2023  hfvienna, author of Jobzilla

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["ANTHROPIC_API_KEY"]

anthropic_client = anthropic.Client(api_key=api_key)


def llm(system_message, user_message):
    completion = anthropic_client.completion(
        model="claude-2",
        max_tokens_to_sample=2000,
        prompt="\n\nHuman:" + system_message + user_message + "\n\nAssistant:",
        temperature=0.0,
    )
    return completion
