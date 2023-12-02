# MIT License
#
# Copyright 2023 auto_anki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


""" gpt_prompting.py """
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import openai
import json


def get_gpt_answers(search_term: str) -> list:
    """
    Given a search term, returns the GPT answers

    :param search_term: the prompt to GPT
    :type: str
    :rtype: list
    :return: list of QA's for the search term.
    """
    load_dotenv()
    API_KEY = os.environ["API_KEY"]
    openai.api_key = API_KEY
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": "Can you create 3 anki cards on topic of" + search_term + "? And also avoid single quotes in the answer."
                " I will provide an example below. Make sure to have contain in the brackets '[]' "
                "[{'Question': 'What do principal components mean?', 'Answer': 'Principal components "
                "are new variables that are constructed as linear combinations or mixtures of the initial "
                "variables.'}, "
                "{'Question': 'What is principal component in PCA?', "
                "'Answer': 'Principal component analysis, or PCA, is a dimensionality reduction method that is "
                "often used to reduce the dimensionality of large data sets, by transforming a large set of "
                "variables into a smaller one that still contains most of the information in the large set.'}, "
                "{'Question': 'What is the principal component theory?', 'Answer': 'PCA is defined as an orthogonal "
                "linear transformation that transforms the data to a new coordinate system such that the greatest "
                "variance by some scalar projection of the data comes to lie on the first coordinate "
                "(called the first principal component), the second greatest variance on the second coordinate, and so on.'}] "
            },
        ],
    )
    res = completion.choices[0].message.content
    res_json = res.replace("'", '"')
    res_list = json.loads(res_json)

    return res_list
