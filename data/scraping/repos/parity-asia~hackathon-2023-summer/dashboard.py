from decouple import config
import logging
import pandas as pd
from fuzzywuzzy import process

from typing import List
from fastapi import APIRouter, Request, Body, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


from backend.api.v1.models.openai import ChatBase, ChatDB
from services.dp_openai import OpenAIService


router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
    responses={404: {"description": "Not found"}},
)




# @router.post(
#         "/", 
#         response_description="Process User Input", 
#         responses={404: {"description": "Not found"}}
# )
# async def process_user_input(request: Request, input_prompt: ChatBase = Body(...)):
#     try:

#         # Extract time and date from input
#         time_intervals = OpenAIService.extract_info_from_prompt(input_prompt.prompt)
#         #symbol_list = get_symbol_fuzzy(symbol_names)

#         #logging.debug(f'time_symbol: {time_intervals, symbol_list}')
#         print(f'time_symbol: {time_intervals}')
#         # Get data from binance API
#         # binance_data = binance.get_historical_price(
#         #     symbol_list,
#         #     'USDT', 
#         #     '8h', 
#         #     time_intervals,
#         # )
 
#         # input_prompt.binance_data = binance_data.to_dict(orient="records")
#         # input_prompt.news_data = news.get_top_headlines(f'{symbol_list[0]}')

#         js = jsonable_encoder(input_prompt)

#         result = await request.app.mongodb["polardash"].insert_one(js)
#         inserted_id = result.inserted_id

#         created_prompt = await request.app.mongodb["polardash"].find_one(
#             {"_id": str(result.inserted_id)},
#             projection={"_id": 1, "prompt": 1 },
#         )

#         if created_prompt is None:
#             raise HTTPException(status_code=404, detail="Chat not found")

#         return {"id": inserted_id, **created_prompt}

#     except HTTPException as e:
#         raise e

#     except (TypeError, AttributeError, ValueError) as e:
#         raise HTTPException(status_code=400, detail="Invalid input data")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Internal server error")



# @router.get(
#         "/", 
#         response_description="List chat history",
#         responses={404: {"description": "Not found"}}
# )
# async def list_chat_history(request: Request, skip: int = 0, limit: int = 100) -> List[ChatDB]:
#     try:
#         projection = {"_id": 1, "prompt": 1, "binance_data": 1, "news_data": 1 }
#         full_query = request.app.mongodb['polardash'].find({}, projection).sort([('_id', 1)]).skip(skip).limit(limit)

#         results = [ChatDB(**chat) async for chat in full_query]

#         return results

#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": str(e)})





