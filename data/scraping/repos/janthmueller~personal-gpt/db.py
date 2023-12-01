from langchain.vectorstores import Chroma
import chromadb
import argparse
import os
from shutil import rmtree


parser = argparse.ArgumentParser(description="DB and collection management.")
parser.add_argument(
    "-p",
    "--persist-dir",
    type=str,
    default="./db",
    help="Directory to persist the database in.",
)
parser.add_argument("-d", "--delete", action="store_true",
                    help="Delete the database.")
parser.add_argument(
    "-c",
    "--create-collections",
    nargs="+",
    default=[],
    help="Name/s of the collection/s to use.",
)
parser.add_argument(
    "-dc",
    "--delete-collections",
    nargs="+",
    default=[],
    help="Name/s of the collection/s to delete.",
)
parser.add_argument(
    "-r",
    "--rename-collections",
    nargs="+",
    default=[],
    help="Rename collection/s. Format: old_name0 new_name0 old_name1 new_name1 ...",
)
parser.add_argument("-l", "--list", action="store_true",
                    help="List all collections.")

parser.set_defaults(delete=False)
parser.set_defaults(list=False)


class DB:
    def __init__(self, persist_dir="./db"):
        self.client = chromadb.PersistentClient(persist_dir)

    def create_collection(self, name):
        self.client.create_collection(name)

    def delete_collection(self, name):
        self.client.delete_collection(name)

    def rename_collection(self, old_name, new_name):
        self._get_collection(old_name).modify(new_name)

    def _get_collection(self, name):
        return self.client.get_collection(name)

    def _get_collections_info(self):
        return self.client._sysdb.get_collections()

    def _get_collection_names(self):
        collections_info = self._get_collections_info()
        return [info["name"] for info in collections_info]

    def get_langchain_collection(self, name, embedder):
        if name in self._get_collection_names():
            return Chroma(
                client=self.client, collection_name=name, embedding_function=embedder
            )
        else:
            raise ValueError(f"Collection {name} does not exist.")


def del_db(persist_dir):
    try:
        rmtree(persist_dir)
    except OSError as e:
        print("Error: %s : %s" % (persist_dir, e.strerror))


if __name__ == "__main__":
    args = parser.parse_args()

    if args.delete:
        if not os.path.exists(args.persist_dir):
            raise ValueError(
                f"Database at {args.persist_dir} does not exist. Cannot delete."
            )
        del_db(args.persist_dir)
        print(f"Deleted database at {args.persist_dir}.")
        exit()

    if os.path.exists(args.persist_dir):
        print(f"Loading database at {args.persist_dir}.")
    else:
        print(f"Creating database at {args.persist_dir}.")

    db = DB(args.persist_dir)
    for collection in args.create_collections:
        db.create_collection(collection)
        print(f"Created collection {collection}.")

    for collection in args.delete_collections:
        db.delete_collection(collection)
        print(f"Deleted collection {collection}.")

    for old_name, new_name in zip(
        args.rename_collections[::2], args.rename_collections[1::2]
    ):
        db.rename_collection(old_name, new_name)
        print(f"Renamed collection {old_name} to {new_name}.")

    if args.list:
        if db._get_collections_info():
            print(f"Collections:")
            for name in db._get_collections_info():
                print(f"{name}")
        else:
            print("No collections.")
