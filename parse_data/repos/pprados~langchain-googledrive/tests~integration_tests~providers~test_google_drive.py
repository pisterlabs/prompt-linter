def test_google_driver() -> None:
    from langchain_googledrive.retrievers import GoogleDriveRetriever

    folder_id = "root"
    retriever = GoogleDriveRetriever(
        num_results=2,
    )
    for doc in retriever.get_relevant_documents("machine learning"):
        print("---")
        print(doc.page_content.strip()[:60] + "...")
    # %%
    retriever = GoogleDriveRetriever(
        template="gdrive-query",  # Search everywhere
        num_results=2,  # But take only 2 documents
    )
    for doc in retriever.get_relevant_documents("machine learning"):
        print("---")
        print(doc.page_content.strip()[:60] + "...")
    # %%
    from langchain import PromptTemplate

    retriever = GoogleDriveRetriever(
        template=PromptTemplate(
            input_variables=["query"],
            # See https://developers.google.com/drive/api/guides/search-files
            template="(fullText contains '{query}') "
            "and mimeType='application/vnd.google-apps.document' "
            "and modifiedTime > '2000-01-01T00:00:00' "
            "and trashed=false",
        ),
        num_results=2,
        # See https://developers.google.com/drive/api/v3/reference/files/list
        includeItemsFromAllDrives=False,
        supportsAllDrives=False,
    )
    for doc in retriever.get_relevant_documents("machine learning"):
        print(f"{doc.metadata['name']}:")
        print("---")
        print(doc.page_content.strip()[:60] + "...")
    # %%
    retriever = GoogleDriveRetriever(
        template="gdrive-mime-type-in-folder",
        folder_id=folder_id,
        mime_type="application/vnd.google-apps.document",  # Only Google Docs
        num_results=2,
        mode="snippets",
        includeItemsFromAllDrives=False,
        supportsAllDrives=False,
    )
    retriever.get_relevant_documents("machine learning")
