#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from openai_helper.dmo import EtlRemoveEmojis


def test_component():

    remove = EtlRemoveEmojis().process
    assert remove

    input_text = 'List 10 things to do today'

    output_text = """Oh, come on, the consequences are :fake_emoji: never that bad. It's all alarmist hype :temp: and hysteria. We just have to throw a little money at it and everything will be alright. :roll_eyes:"""

    actual_result = remove(
        input_text=input_text,
        output_text=output_text)

    print(actual_result)


def main():
    test_component()


if __name__ == '__main__':
    main()
