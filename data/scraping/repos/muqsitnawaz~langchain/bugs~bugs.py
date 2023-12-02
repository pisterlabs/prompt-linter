from langchain.vectorstores import Chroma
from chromadb.config import Settings


def main() -> None:
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=".dbase")
    db = Chroma(client_settings=settings)
    db.persist()


if __name__ == "__main__":
    main()
