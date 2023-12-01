import asyncio
import aiofiles
import csv
from datetime import datetime
import json
from logging import getLogger, NullHandler
import traceback
from typing import List
import sqlite3
import sqlite_vss
import numpy as np
from openai import Embedding

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class VSSLite:
    def __init__(self, openai_apikey: str, connection_str: str="vss.db"):
        self.openai_apikey = openai_apikey
        self.connection_str = connection_str
        self.create_tables()

    def sync(self, future):
        return asyncio.get_event_loop().run_until_complete(future)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.connection_str,
            isolation_level=None
        )
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        return conn

    def create_tables(self):
        conn = self.get_connection()

        try:
            conn.execute("create table if not exists knowledges (id INTEGER primary key, updated_at DATETIME, namespace TEXT, body TEXT, serialized_json TEXT)")
            conn.execute("create virtual table if not exists embeddings using vss0 (body_embedding(1536))")
            conn.commit()
        
        except Exception as ex:
            logger.error(f"Error at VSSEngine.create_tables: {str(ex)}\n{traceback.format_exc()}")
            conn.rollback()
            raise ex

        finally:
            conn.close()

    @staticmethod
    def vector_to_bytes(vector: List[float]) -> bytes:
        return np.asarray(vector).astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_vector(embedding: bytes) -> List[float]:
        return np.frombuffer(embedding, dtype=np.float32).tolist()

    async def acreate_embedding(self, text: str) -> List[float]:
        response = await Embedding.acreate(
            api_key = self.openai_apikey,
            engine="text-embedding-ada-002",
            input=[text]
        )
        return response["data"][0]["embedding"]

    async def aadd(self, body: str, data: dict=None, namespace: str="default") -> int:
        now = datetime.utcnow()
        embedding = await self.acreate_embedding(body)

        conn = self.get_connection()

        try:
            conn.execute(
                "insert into knowledges (updated_at, namespace, body, serialized_json) values (?, ?, ?, ?)",
                (now, namespace, body, json.dumps(data, ensure_ascii=False) if data else "{}")
            )

            last_id = conn.execute("select last_insert_rowid()").fetchone()[0]

            conn.execute(
                "insert into embeddings (rowid, body_embedding) values (?, ?)",
                (last_id, self.vector_to_bytes(embedding))
            )
            
            conn.commit()

            return last_id
        
        except Exception as ex:
            logger.error(f"Error at VSSEngine.add: {str(ex)}\n{traceback.format_exc()}")
            conn.rollback()
            raise ex
            
        finally:
            conn.close()
    
    def add(self, body: str, data: dict=None, namespace: str="default") -> int:
        return self.sync(self.aadd(body, data, namespace))

    async def aupdate(self, id: int, body: str, data: dict=None) -> int:
        now = datetime.utcnow()
        embedding = await self.acreate_embedding(body)

        conn = self.get_connection()

        try:
            current_record_namespace = conn.execute(
                "select namespace from knowledges where id = ?",
                (id, )
            ).fetchone()[0]

            # Delete and add because virtual table doesn't support update
            conn.execute("delete from knowledges where id = ?", (id, ))
            conn.execute("delete from embeddings where rowid = ?", (id, ))
            conn.execute(
                "insert into knowledges (updated_at, namespace, body, serialized_json) values (?, ?, ?, ?)",
                (now, current_record_namespace, body, json.dumps(data, ensure_ascii=False) if data else "{}")
            )
            last_id = conn.execute("select last_insert_rowid()").fetchone()[0]
            conn.execute(
                "insert into embeddings (rowid, body_embedding) values (?, ?)",
                (last_id, self.vector_to_bytes(embedding))
            )

            conn.commit()

            return last_id
        
        except Exception as ex:
            logger.error(f"Error at VSSEngine.update: {str(ex)}\n{traceback.format_exc()}")
            conn.rollback()
            raise ex

        finally:
            conn.close()

    def update(self, id: int, body: str, data: dict=None) -> int:
        return self.sync(self.aupdate(id, body, data))

    async def adelete(self, id: int):
        conn = self.get_connection()

        try:
            conn.execute("delete from knowledges where id = ?", (id, ))
            conn.execute("delete from embeddings where rowid = ?", (id, ))
            conn.commit()

        except Exception as ex:
            logger.error(f"Error at VSSEngine.delete: {str(ex)}\n{traceback.format_exc()}")
            conn.rollback()
            raise ex

        finally:
            conn.close()

    def delete(self, id: int):
        self.sync(self.adelete(id))

    async def adelete_all(self):
        conn = self.get_connection()

        try:
            conn.execute("delete from knowledges")
            conn.execute("delete from embeddings")
            conn.commit()

        except Exception as ex:
            logger.error(f"Error at VSSEngine.delete_all: {str(ex)}\n{traceback.format_exc()}")
            conn.rollback()
            raise ex

        finally:
            conn.close()

    def delete_all(self):
        self.sync(self.adelete_all())

    async def aget(self, id: int) -> dict:
        conn = self.get_connection()

        try:
            record = conn.execute("""
                select knowledges.id, knowledges.updated_at, knowledges.namespace, knowledges.body, knowledges.serialized_json, embeddings.body_embedding
                from knowledges
                join embeddings on knowledges.id = embeddings.rowid
                where knowledges.id = ?
            """,
            (id, )
            ).fetchone()

            if record:
                return {
                    "id": record[0],
                    "updated_at": record[1],
                    "namespace": record[2],
                    "body": record[3],
                    "data": json.loads(record[4]),
                    "body_embedding": self.bytes_to_vector(record[5])
                }

        except Exception as ex:
            logger.error(f"Error at VSSEngine.get: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            conn.close()

    def get(self, id: int) -> dict:
        return self.sync(self.aget(id))

    async def asearch(self, query: str, count: int=1, namespace: str="default") -> List[dict]:
        query_embedding = await self.acreate_embedding(query)

        conn = self.get_connection()

        try:
            records = conn.execute("""
                select knowledges.id, knowledges.updated_at, knowledges.namespace, knowledges.body, knowledges.serialized_json, embeddings.distance
                from knowledges
                join embeddings on knowledges.id = embeddings.rowid
                where vss_search(embeddings.body_embedding, vss_search_params(?, 10)) and knowledges.namespace = ?
                order by embeddings.distance
                limit ?""",
            (self.vector_to_bytes(query_embedding), namespace, count)
            ).fetchall()

            ret = []
            for record in records:
                ret.append({
                    "id": record[0],
                    "updated_at": record[1],
                    "namespace": record[2],
                    "body": record[3],
                    "data": json.loads(record[4]),
                    "distance": record[5]
                })

            return ret

        except Exception as ex:
            logger.error(f"Error at VSSEngine.search: {str(ex)}\n{traceback.format_exc()}")
            raise ex
        
        finally:
            conn.close()

    def search(self, query: str, count: int=1, namespace: str="default") -> List[dict]:
        return self.sync(self.asearch(query, count, namespace))

    async def aload_records_as_json(self, path) -> List[dict]:
        async with aiofiles.open(path, mode="r", newline="") as file:
            content = await file.read()
            try:
                return json.loads(content)["records"]
            except Exception as ex:
                csv_lines = content.split("\n")
                reader = csv.DictReader(csv_lines)
                return [dict(r) for r in reader]

    async def aimport_file(self, path: str, body_key: str="body", namespace: str="default"):
        records = await self.aload_records_as_json(path)

        ret = {"ids": [], "errors": []}
        for r in records:
            try:
                if "id" in r:
                    ret["ids"].append(await self.aupdate(r["id"], r[body_key], r))
                else:
                    ret["ids"].append(await self.aadd(r[body_key], r, namespace))
            except Exception as ex:
                ret["errors"].append({"message": str(ex), "record": r})

        return ret

    def import_file(self, path: str, body_key: str="body", namespace: str="default"):
        return self.sync(self.aimport_file(path, body_key, namespace))
