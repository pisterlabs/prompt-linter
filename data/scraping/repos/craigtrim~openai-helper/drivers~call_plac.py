#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Plac/Terminal Test """


from baseblock import Enforcer

from openai_helper import call2


def extract_output(input_text: str):

    if type(input_text) == tuple:
        input_text = input_text[0]

    Enforcer.is_str(input_text)

    print(call2(input_text))


def main(input_text):
    from drivers import IntegrationWrapper
    wrapper = IntegrationWrapper()

    wrapper.call(extract_output, input_text)

    wrapper.deconstruct_env()


if __name__ == '__main__':
    import plac

    plac.call(main)
