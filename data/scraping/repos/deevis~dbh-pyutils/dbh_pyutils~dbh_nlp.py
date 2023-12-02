import nltk
nltk.download('punkt')
from langchain.text_splitter import NLTKTextSplitter

def chunkify(text, split_size: int = 400, min_chunk_size: int = 20):
    """Returns a list of strings of length `size`."""
    if text is None or len(text) < split_size:
        return [text]
    # Use NLTKTextSplitter to split text into sentences
    text_splitter = NLTKTextSplitter()
    sentences = text_splitter.split_text(text)    
    # print(f'\nType: {type(docs)}')
    # print(f'\nLen: {len(docs)}')
    
    split_out = [x.split("\n\n") for x in sentences]
    split_out = [item for sublist in split_out for item in sublist]
    # print(f'\nsentences Type: {type(sentences)}')
    # print(f'\nsentencesLen: {len(sentences)}')
    # for s2 in sentences:
    #     print(f'\ns2[{len(s2)}]: {s2}')

    chunks = []
    chunk = ""
    for s2 in split_out:
        if len(s2) == 0:
            continue
        chunk += s2 + "\n"
        if len(chunk) > split_size:
            chunks.append(chunk)
            chunk = ""
    if len(chunk) > 0:
        if len(chunk) < min_chunk_size and len(chunks) > 0:
            # just add it to the last chunk rather than adding it as a new chunk
            chunks[-1] += f'{chunk}\n'
        else:
            chunks.append(chunk)
    return chunks
