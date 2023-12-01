# https://en.wikipedia.org/wiki/Giacomo_Puccini


def test_puccini_page_fully():
    from langchain.document_loaders import WikipediaLoader

    loader = WikipediaLoader(query="Giacomo Puccini", load_max_docs=1, doc_content_chars_max=100_000)
    doc = loader.load()[0]
    assert (
        "was an Italian composer known primarily for his operas. Regarded as the greatest and most successful proponent"
        " of Italian opera after Verdi, he was descended from a long line of composers, stemming from the late-Baroque"
        " era. Though his early work was firmly rooted in traditional late-19th-century Romantic Italian opera, he"
        " later developed his work in the realistic verismo style, of which he became one of the leading exponents."
        in doc.page_content
    )
    assert (
        "Turandot, libretto by Renato Simoni and Giuseppe Adami (in three acts – incomplete at the time of Puccini's"
        " death, completed by Franco Alfano: premiered at La Scala, 25 April 1926)"
        in doc.page_content
    )
