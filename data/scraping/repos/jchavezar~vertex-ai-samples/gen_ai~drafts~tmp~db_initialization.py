##### Global Variables
#%%
from credentials import *

##### Create Cloud SQL Database Table (transcripts no embeddings)
#%%
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector
import numpy as np
from pgvector.asyncpg import register_vector
async def main():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)

        await conn.execute("DROP TABLE IF EXISTS video_metadata")
        # Create the `video_metadata` table.
        await conn.execute(
            """CREATE TABLE video_metadata(
                                transcription VARCHAR(52000),
                                class VARCHAR(10),
                                summary VARCHAR(1000),
                                video_link VARCHAR(100),
                                snippet_link VARCHAR(100))"""
        )
        await conn.close()
await main()  # type: ignore


##### Testing Cloud SQL Query
# %%
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector
import numpy as np
from pgvector.asyncpg import register_vector

async def main():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )
        x=await conn.fetch(
            """SELECT * from video_metadata"""
        )
        print(len(x))
        for n,i in enumerate(x):
            if "Messi" in i["transcription"]:
                print(i["transcription"])
                print(i["video_link"])
                print(n)
        await conn.close()
await main() 


##### Create Cloud SQL Database Table (embeddings)
# %%
import pandas as pd
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector
import numpy as np
from pgvector.asyncpg import register_vector

df = pd.read_csv("emb.csv")
df2 = df.copy()
df2["embedding"] = df2["embedding"].apply(lambda x: x.strip("][").split(","))

async def main():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        await conn.execute("DROP TABLE IF EXISTS video_embeddings")
        # Create the `video_embeddings` table to store vector embeddings.
        await conn.execute(
            """CREATE TABLE video_embeddings(
                                summary VARCHAR(1024),
                                frame_link VARCHAR(100),
                                video_link VARCHAR(100),
                                embedding vector(1408))"""
        )

        # Store all the generated embeddings back into the database.
        for index, row in df2.iterrows():
            print(np.array(row["embedding"]))
            await conn.execute(
                "INSERT INTO video_embeddings (summary, frame_link, video_link, embedding) VALUES ($1, $2, $3, $4)",
                row["summary"],
                row["frame_link"],
                row["video_link"],
                np.array(row["embedding"]),
            )

        x = await conn.fetch("SELECT * FROM video_embeddings")
        for i in x:
           print("te")
           print(i)
        await conn.close()
await main()  # type: ignore


##### Insert Items into DB Table
#%%
import pandas as pd
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector
import numpy as np
from pgvector.asyncpg import register_vector

df = pd.read_csv("emb.csv")
df2 = df.copy()
df2["embedding"] = df2["embedding"].apply(lambda x: x.strip("][").split(","))

async def main():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        # Create the `video_embeddings` table to store vector embeddings.

        # Store all the generated embeddings back into the database.
        for index, row in df2.iterrows():
            print(np.array(row["embedding"]))
            await conn.execute(
                "INSERT INTO video_embeddings (summary, frame_link, video_link, embedding) VALUES ($1, $2, $3, $4)",
                row["summary"],
                row["frame_link"],
                row["video_link"],
                np.array(row["embedding"]),
            )

        x = await conn.fetch("SELECT * FROM video_embeddings")
        for i in x:
           print("te")
           print(i)
        await conn.close()
await main()  # type: ignore




#Query
#%%
# @markdown Enter a short description of the toy to search for within a specified price range:
toy = "canada"  # @param {type:"string"}

from langchain.embeddings import VertexAIEmbeddings
from google.cloud import aiplatform
from pgvector.asyncpg import register_vector
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector
import sys
sys.path.append("..")
from ai import multimodal as mm

matches = []

qe = mm.get_embedding(text=toy).text_embedding

async def main():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await register_vector(conn)
        similarity_threshold = 0.001
        num_matches = 50

        # Find similar products to the query using cosine similarity search
        # over all vector embeddings. This new feature is provided by `pgvector`.
        results = await conn.fetch(
            """
                            WITH vector_matches AS (
                              SELECT summary, frame_link, video_link, 1 - (embedding <=> $1) AS similarity
                              FROM video_embeddings
                              WHERE 1 - (embedding <=> $1) > $2
                              ORDER BY similarity DESC
                              LIMIT $3
                            )
                            SELECT * FROM vector_matches
                            """,
            qe,
            similarity_threshold,
            num_matches,
        )

        if len(results) == 0:
            raise Exception("Did not find any results. Adjust the query parameters.")

        for r in results:
            # Collect the description for all the matched similar toy products.
            matches.append(
                {
                    "summary": r["summary"],
                    "frame_link": r["frame_link"],
                    "video_link": r["video_link"],
                    "similarity": r["similarity"]
                }
            )

        await conn.close()


# Run the SQL commands now.
await main()  # type: ignore

# Show the results for similar products that matched the user query.
matches = pd.DataFrame(matches)
matches.head(10)


# %%
