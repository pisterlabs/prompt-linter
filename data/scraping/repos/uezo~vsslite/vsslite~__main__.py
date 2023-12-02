# Check SQLite version to satisfy ChromaDB requirements
import sqlite3
if sqlite3.sqlite_version_info < (3, 35, 0):
    import sys
    try:
        __import__("pysqlite3")
    except ImportError:
        import subprocess
        print("Start installing additional dependencies for ChromaDB ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pysqlite3-binary"]
        )
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import argparse
import os
import uvicorn
from vsslite import LangChainVSSLiteServer

apikey = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser(description="VSSLite usage")
parser.add_argument("--host", type=str, default="127.0.0.1", required=False, help="hostname or ipaddress")
parser.add_argument("--port", type=int, default="8000", required=False, help="port number")
parser.add_argument("--apikey", type=str, default=apikey, required=False, help="OpenAI API Key")
parser.add_argument("--dir", type=str, default="./vectorstore", required=False, help="Data persist directory")
parser.add_argument("--chunksize", type=int, default=500, required=False, help="Chunk size")
parser.add_argument("--chunkoverlap", type=int, default=0, required=False, help="Chunk overlap")
parser.add_argument("--vectorstore", type=str, default="chromadb", required=False, help="Chunk overlap")
args = parser.parse_args()


if args.vectorstore == "sqlite":
    from vsslite import VSSLiteServer
    vss = VSSLiteServer(
        openai_apikey=args.apikey,
        connection_str=args.dir
    )

else:
    from vsslite import LangChainVSSLiteServer
    vss = LangChainVSSLiteServer(
        apikey=args.apikey,
        persist_directory=args.dir,
        chunk_size=args.chunksize,
        chunk_overlap=args.chunkoverlap
    )

uvicorn.run(vss.app, host=args.host, port=args.port)
