#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Integration Test """


from pprint import pprint

from baseblock import Enforcer

from openai_helper.bp import OpenAIChatCompletion
from openai_helper import chat


def test_completion_1():

    bp = OpenAIChatCompletion()
    assert bp

    d_result = bp.run(
        input_prompt='You are a librarian in New York City.',
        messages=[
            'What meal do you treasure the most?',
            'I was partial to ice cream.',
            'Do you have a favorite food?',
            'I was partial to ice cream.',
            'Which cuisine are you passionate about and why?',
            'I was partial to ice cream.',
            'what is your favorite food.',
        ])

    pprint(d_result)
    Enforcer.keys(d_result, 'input', 'output')


def main():
    from drivers import IntegrationWrapper
    wrapper = IntegrationWrapper()

    wrapper.call(test_completion_1)

    wrapper.deconstruct_env()


if __name__ == '__main__':
    main()
