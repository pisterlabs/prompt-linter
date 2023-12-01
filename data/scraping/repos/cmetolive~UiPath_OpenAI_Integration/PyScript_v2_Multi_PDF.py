def AnalyzeDataset(user_query: str):
    from langchain.document_loaders import UnstructuredPDFLoader
    from langchain.indexes import VectorstoreIndexCreator

    import os
    os.environ["OPENAI_API_KEY"]="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"

    # Dataset path
    pdf_folder_path = 'D:\Datasets\FinData\Tesla'
    os.listdir(pdf_folder_path)

    # Load multiple PDF data files.
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
    #loaders

    index = VectorstoreIndexCreator().from_loaders(loaders)
    #index

    # index.query('What was the main topic of the address?')
    # ans = index.query_with_sources('In establishing the Revenue and Adjusted EBITDA milestones, what were the variety of factors that were carefully considered by the Board?')
    # print(str(ans))
    ans = index.query_with_sources(user_query)
    # print(str(ans['answer']) + '\n\n\n' + "Source File Name: " + str(ans['sources']))
    return str(ans['answer']) + "Source File Name: " + str(ans['sources'])

# def main():
#    callRet=AnalyzeDataset("In establishing the Revenue and Adjusted EBITDA milestones, what were the variety of factors that were carefully considered by the Board?")
#    print("After Ret: " + callRet)
#    
# if __name__ == '__main__':
#    main()







