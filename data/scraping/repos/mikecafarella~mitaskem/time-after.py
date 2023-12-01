# import sys
# srcpath = '/Users/orm/repos/mitaskem/src'
# askempath = '/Users/orm/repos/mitaskem'
# sys.path.append(askempath)
# sys.path.append(srcpath)

import os
import openai
from fastapi import UploadFile
from api.routers.annotation import upload_file_annotate 
import asyncio
import json
import io


async def test():
    filename = os.path.dirname(__file__) + '/../../resources/models/Bucky/bucky.txt'
    with open(filename, 'rb') as f:
        c = f.read()

    f = io.BytesIO(c)
    upfile = UploadFile(
        filename=os.path.basename(filename),
        file=f
    )
    res = await upload_file_annotate(gpt_key=openai.api_key,file=upfile)

    return res

res = asyncio.run(test())
print('results:')
print('-----')
print(res)

print('result:', res.json())

