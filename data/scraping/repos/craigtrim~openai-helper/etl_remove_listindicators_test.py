#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from openai_helper.dmo import EtlRemoveListIndicators


def test_component():
    dmo = EtlRemoveListIndicators()
    assert dmo

    input_text = 'List 10 things to do today'

    output_text = """
        1. wake up
        2. eat breakfast
        3. go to work
        4. eat lunch
        5. come home from work
        6. eat dinner
        7. spend time with family/friends
        8. read
        9. watch tv
        10. go to bed
    """

    actual_result = dmo.process(
        input_text=input_text,
        output_text=output_text)

    expected_result = """
wake up
eat breakfast
go to work
eat lunch
come home from work
eat dinner
spend time with family/friends
read
watch tv
go to bed
    """.strip()

    if actual_result != expected_result:
        print(actual_result)

    assert actual_result == expected_result


def main():
    test_component()


if __name__ == '__main__':
    main()
