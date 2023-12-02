from langchain.text_splitter import RecursiveCharacterTextSplitter
from diskio import page_content


def test_split(content: str):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a tiny chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=400,
        length_function=len,
    )

    splits = text_splitter.create_documents([content])
    print('num splits: {}'.format(len(splits)))

    for split in splits:
        print(split)


if __name__ == '__main__':
    path = '../data/videos/youtube/@hubermanlab/Adderall, Stimulants & Modafinil for ADHD: Short- & Long-Term Effects | Huberman Lab Podcast.json'
    content = page_content(path)

    test_split(content)
