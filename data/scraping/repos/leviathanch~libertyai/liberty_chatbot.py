from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Sequence,
    List,
    Union,
)

import langchain

from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import PostgresChatMessageHistory
from langchain.chains import ConversationChain

from LibertyAI.liberty_chain import LibertyChain
from LibertyAI.liberty_config import get_configuration
from LibertyAI.liberty_agent import (
    get_zero_shot_agent,
    get_vector_db
)

def initialize_chatbot(**kwargs: Any) -> LibertyChain:

    main_llm = kwargs['llm']
    main_emb = kwargs['emb']
    try:
        sqlstring = kwargs['sqlstring']
        user_mail = kwargs['email']
    except:
        history = None
    else:
        history = PostgresChatMessageHistory(
            connection_string=sqlstring,
            session_id="history_"+user_mail
        )

    conv_mem = ConversationBufferWindowMemory(
        ai_prefix = "LibertyAI",
        k = 5,
    )

    last_few = []
    if history:
        current_dialog = {}
        for message in history.messages:
            if type(message) == langchain.schema.HumanMessage:
                if 'input' not in current_dialog:
                    current_dialog['input'] = message.content
            if type(message) == langchain.schema.AIMessage:
                if 'output' not in current_dialog:
                    current_dialog['output'] = message.content
            if 'input' in current_dialog and 'output' in current_dialog:
                conv_mem.save_context(
                    inputs = {"Human": current_dialog['input']},
                    outputs = {"LibertyAI": current_dialog['output']}
                )
                current_dialog = {}

    #sum_mem = ConversationSummaryMemory(
    #    ai_prefix = "LibertyAI",
    #    llm = main_llm,
    #)

    vecdb = get_vector_db()

    chain = LibertyChain(
        #summary = sum_mem,
        summary = None,
        memory = conv_mem,
        pghistory = history,
        llm = main_llm,
        mrkl = get_zero_shot_agent( main_llm ),
        verbose = True,
        user_name = kwargs['name'],
        user_mail = kwargs['email'],
        embeddings = main_emb,
        vectordb = vecdb,
    );

    return chain
