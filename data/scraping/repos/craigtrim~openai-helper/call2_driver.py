#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Integration Test """


from functools import partial

from openai_helper import call2


def test_call2(input_prompt: str):

    result = call2(input_prompt)
    print(f'Result: {result}')
    assert result


def main(input_prompt):
    from drivers import IntegrationWrapper
    wrapper = IntegrationWrapper()

    service_call = partial(test_call2, input_prompt=input_prompt)
    wrapper.call(service_call)

    wrapper.deconstruct_env()


if __name__ == '__main__':
    import plac

    plac.call(main)
