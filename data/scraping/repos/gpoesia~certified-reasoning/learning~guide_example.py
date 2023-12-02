#!/usr/bin/env python3

import openai
from synchromesh import predict_constrained
from language_model import OpenAIModel

import regex
import completion



class CountingGuide:
    '''Completion engine implementing a "counting guide": given
    a starting number s and a step size k, each guided block
    will be forced to contain s + i*k for i = 0, 1, 2, ...

    Outside guided blocks, the model can output anything.

    One could also make a more elaborate extension of this where
    one kind of guided block resets the initial parameters to whatever
    the model chooses (e.g. [[count:103 by 13]]) and another kind of block
    always outputs the next number in the sequence (e.g., [[next:116]]).
    '''
    def __init__(self, starting_number, step_size,
                 start_marker='[[', end_marker=']]'):
        self.starting_number = starting_number
        self.step_size = step_size
        self.start_marker = start_marker
        self.end_marker = end_marker

    def complete(self, prefix: str):
        if not prefix.endswith(self.start_marker):
            # We're outside a guided block. Let the model generate anything
            # until it generates the start marker.
            # Note that now we only need to worry about what happens once the
            # model has generated something ending with the start marker.
            # because of the contract between Synchromesh and completion engines.
            return regex.compile(completion.regex_not_containing(self.start_marker) +
                                 regex.escape(self.start_marker))

        # We're in a newly open guided block.
        # Let's first count how many previous guided blocks were there:
        i = prefix.count(self.start_marker)

        # Only let the model return the next correct number, followed by the
        # end marker.
        return regex.compile(str(self.starting_number + self.step_size * i) +
                             regex.escape(self.end_marker))

    def is_complete(self, prefix: str) -> bool:
        # Here we decide when the model is done with its response.
        # Let's stop generation after a few guided blocks.
        return prefix.count(self.end_marker) >= 3


def main():
    prompt = ("A very important and very long sequence in the science of natural numbers is the following: " +
              "it starts with [[103]], then goes to [[116]], followed")

    lm = OpenAIModel('text-curie-001', prompt)
    response = lm.predict_unconstrained('', 100)

    guide = CountingGuide(103 + 2*13, 13)

    response_with_guide = predict_constrained(guide, lm, batch_size=100, stop_tokens=['\n'])

    print('Response without guide:', response)
    print('Response with guide:', response_with_guide)


if __name__ == '__main__':
    main()
