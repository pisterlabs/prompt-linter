# AidBot - Telegram bot project for finding volunteer help using semantic search
# Copyright (C) 2023
# Anastasia Mayorova aka EternityRei  <anastasiamayorova2003@gmail.com>
#    Andrey Vlasenko aka    chelokot   <andrey.vlasenko.work@gmail.com>
from typing import Type

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or any later version. This
# program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details. You should have received a copy of the GNU General Public License along with this program. If not,
# see <https://www.gnu.org/licenses/>.

import requests
from bs4 import BeautifulSoup as bs
import json
from src.database.ProposalsTable import ProposalsTable
from src.embeddings.EmbeddingUtils import EmbeddingUtils
from src.embeddings.OpenAITextEmbedder import OpenAITextEmbedder

from src.database.data_types.UahelpersProposal import UahelpersProposal

from src.database.data_types.ColumnNames import ColumnNames


class UahelpersManager:
    def __init__(self, table_name: str, openai_api_key: str, embedding_type: Type):
        self.db = ProposalsTable()
        self.db.create_table_or_add_columns(table_name)
        self.embedding_type = embedding_type
        self.ai = OpenAITextEmbedder(openai_api_key, embedding_type)
    
    def parse(self):
        has_more = True
        skip_value = 0

        while has_more:
            url = f'https://uahelpers.com/api/volunteers/search?location=&category=&skip={skip_value}'
            response = requests.request("GET", url)

            parsed_text = bs(response.text, 'lxml')
            full_json_str = json.loads(parsed_text.find('p').text)

            has_more = full_json_str['hasMore']
            result = full_json_str['result']

            for proposition in result:
                try:
                    proposition_json = json.dumps(proposition)
                    dict_json = json.loads(proposition_json)

                    proposal = UahelpersProposal(
                        characteristics={
                            ColumnNames.proposal_name:          dict_json['name'],
                            ColumnNames.description:            dict_json['description'],
                            ColumnNames.proposal_contact:       dict_json['contact'],
                            ColumnNames.proposal_comment:       dict_json['comment'],
                            ColumnNames.proposal_location: ', '.join(dict_json['location']),
                            ColumnNames.proposal_services: ', '.join(dict_json['services']),
                            ColumnNames.proposal_date_time:     dict_json['date'],
                        },
                        embedder=self.ai,
                    )

                    if not EmbeddingUtils.check_is_request(self.ai, proposal.embedding):
                        self.db.add(proposal)
                except Exception as e:
                    print(e)
                    continue
            skip_value += len(result)
