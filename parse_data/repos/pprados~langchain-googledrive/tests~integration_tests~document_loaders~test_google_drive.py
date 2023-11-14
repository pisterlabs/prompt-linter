from langchain_googledrive.document_loaders import GoogleDriveLoader


def test_google_drive() -> None:
    folder_id = "root"
    # folder_id='1yucgL9WGgWZdM1TOuKkeghlPizuzMYb6'
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=False,
        num_results=2,  # Maximum number of file to load
    )
    for doc in loader.load():
        print("---")
        print(doc.page_content.strip()[:60] + "...")

    # %%
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=False,
        template="gdrive-query",  # Default template to use
        query="machine learning",
        num_results=2,  # Maximum number of file to load
        supportsAllDrives=False,  # GDrive `list()` parameter
    )
    for doc in loader.load():
        print("---")
        print(doc.page_content.strip()[:60] + "...")
    # %%
    from langchain.prompts.prompt import PromptTemplate

    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=False,
        template=PromptTemplate(
            input_variables=["query", "query_name"],
            template="fullText contains '{query}' and name contains "
            "'{query_name}' and trashed=false",
        ),  # Default template to use
        query="machine learning",
        query_name="ML",
        num_results=2,  # Maximum number of file to load
    )
    for doc in loader.load():
        print("---")
        print(doc.page_content.strip()[:60] + "...")
    # %%
    loader = GoogleDriveLoader(
        template="gdrive-mime-type",
        mime_type="application/vnd.google-apps.presentation",  # Only GSlide files
        gslide_mode="slide",
        num_results=2,  # Maximum number of file to load
    )
    for doc in loader.load():
        print("---")
        print(doc.page_content.strip()[:60] + "...")
    # %%
    loader = GoogleDriveLoader(
        template="gdrive-mime-type",
        mime_type="application/vnd.google-apps.spreadsheet",  # Only GSheet files
        gsheet_mode="elements",
        num_results=2,  # Maximum number of file to load
    )
    for doc in loader.load():
        print("---")
        print(doc.page_content.strip()[:60] + "...")
    # %%
    import os

    loader = GoogleDriveLoader(
        gdrive_api_file=os.environ["GOOGLE_ACCOUNT_FILE"],
        num_results=2,
        template="gdrive-query",
        filter=lambda search, file: "#test" not in file.get("description", ""),
        query="machine learning",
        supportsAllDrives=False,
    )
    for doc in loader.load():
        print("---")
        print(doc.page_content.strip()[:60] + "...")
