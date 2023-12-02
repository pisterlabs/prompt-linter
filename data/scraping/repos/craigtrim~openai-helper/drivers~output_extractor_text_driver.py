#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Integration Test """


from pprint import pprint

from baseblock import Enforcer

from openai_helper import call2


def extract_output(input_text: str):

    if type(input_text) == tuple:
        input_text = input_text[0]

    Enforcer.is_str(input_text)

    print(call2(input_text))

    # nothing to really test; this is an observation case


def main():
    from drivers import IntegrationWrapper
    wrapper = IntegrationWrapper()

    wrapper.call(extract_output, 'List 10 things to do today')

    wrapper.deconstruct_env()


if __name__ == '__main__':
    main()
