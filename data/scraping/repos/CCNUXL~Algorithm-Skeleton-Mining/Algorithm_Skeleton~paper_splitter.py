from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def splitter(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    # print(f"加载完毕，共加载{len(pages)}页PDF文件")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        length_function=len
    )

    texts = text_splitter.split_documents(pages)
    # print(f"文章共切分为{len(texts)}段")
    # for i in range(0, len(texts)):
    #     print(f"段{i}")
    #     print(texts[i].page_content)
    return texts


if __name__ == "__main__":
    pdf_path = "/Users/xueliang/Desktop/硕士毕业论文/毕业论文参考文献/Distilling Model/2212.00193.pdf"
    result = splitter(pdf_path)
    print(result[0].page_content)
