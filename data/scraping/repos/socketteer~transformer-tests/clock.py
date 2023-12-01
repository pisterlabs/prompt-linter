import time
import openai

lob = '''The universe (which others call the Library) is composed of an indefinite,
perhaps infinite number of hexagonal galleries. In the center of each gallery
is a ventilation shaft, bounded by a low railing. From any hexagon one can
see the floors above and below-one after another, endlessly. The arrange­
ment of the galleries is always the same: Twenty bookshelves, five to each
side, line four of the hexagon's six sides; the height of the bookshelves, floor
to ceiling, is hardly greater than the height of a normal librarian. One of the
hexagon's free sides opens onto a narrow sort of vestibule, which in turn
opens onto another gallery, identical to the first-identical in fact to all.
To the left and right of the vestibule are two tiny compartments. One is
for sleeping, upright; the other, for satisfying one's physical necessities.
Through this space, too, there passes a spiral staircase, which winds upward
and downward into the remotest distance. In the vestibule there is a mirror,
which faithfully duplicates appearances. Men often infer from this mirror
that the Library is not infinite-if it were, what need would there be for that
illusory replication? I prefer to dream that burnished surfaces are a figura­
tion and promise of the infinite ... Light is provided by certain spherical
fruits that bear the name "bulbs." There are two of these bulbs in each hexa­
gon, set crosswise. The light they give is insufficient, and unceasing.
Like all the men of the Library, in my younger days I traveled; I have
journeyed in quest of a book, perhaps the catalog of catalogs. Now that my
eyes can hardly make out what I myself have written, I am preparing to die,
a few leagues from the hexagon where I was born. When I am dead, com­
passionate hands will throw me over the railing; my tomb will be the un­
fathomable air, my body will sink for ages, and will decay and dissolve in the
wind engendered by my fall, which shall be infinite.'''

def elapsed(prompt=None, prompt_length=100, continuation_length=0, engine='ada'):
    print('---------------')
    if prompt is None:
        prompt = lob[:prompt_length]
    else:
        print('prompt: ', prompt)
        prompt_length = len(prompt)

    start_time = time.time()
    rsp = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=continuation_length,
        echo=True,
        logprobs=0,
        timeout=15)
    elapsed_seconds = time.time() - start_time
    print('engine:', engine)
    print('prompt length:', prompt_length)
    print('continuation length:', continuation_length)
    print('total chars:', prompt_length + continuation_length)
    print('second elapsed:', elapsed_seconds)

elapsed(prompt_length=1, continuation_length=100, engine='ada')
elapsed(prompt_length=1, continuation_length=1000, engine='ada')
elapsed(prompt_length=100, continuation_length=0, engine='ada')
elapsed(prompt_length=1000, continuation_length=0, engine='ada')
elapsed(prompt_length=100, continuation_length=100, engine='ada')

elapsed(prompt_length=1, continuation_length=100, engine='babbage')
elapsed(prompt_length=1, continuation_length=1000, engine='babbage')
elapsed(prompt_length=100, continuation_length=0, engine='babbage')
elapsed(prompt_length=1000, continuation_length=0, engine='babbage')
elapsed(prompt_length=100, continuation_length=100, engine='babbage')

elapsed(prompt_length=1, continuation_length=100, engine='curie')
elapsed(prompt_length=1, continuation_length=1000, engine='curie')
elapsed(prompt_length=100, continuation_length=0, engine='curie')
elapsed(prompt_length=1000, continuation_length=0, engine='curie')
elapsed(prompt_length=100, continuation_length=100, engine='curie')

elapsed(prompt_length=1, continuation_length=100, engine='davinci')
elapsed(prompt_length=1, continuation_length=1000, engine='davinci')
elapsed(prompt_length=100, continuation_length=0, engine='davinci')
elapsed(prompt_length=1000, continuation_length=0, engine='davinci')
elapsed(prompt_length=100, continuation_length=100, engine='davinci')