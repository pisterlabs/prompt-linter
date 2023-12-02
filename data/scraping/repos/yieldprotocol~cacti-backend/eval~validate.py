import re
import enum
import argparse
import random
import collections
import copy
import uuid
from typing import Any, Dict, Generator, List, Optional, Union, Literal, TypedDict, Callable, Iterable

import pandas as pd
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage

from integrations.center import (
    NFTCollection, NFTAsset, NFTCollectionAssets, NFTAssetTraits, NFTAssetTraitValue,
    NFTCollectionTraits, NFTCollectionTrait, NFTCollectionTraitValue,
)
from chat.base import ChatHistory, ChatMessage
import chat
import config
import utils.timing as timing
from utils.common import widget_subset, get_user_info
from utils.evaluation import (
    Conversation, Message,
    stream_to_str,
    handle_empty_params,
)
from tools.index_widget import (
    StreamingListContainer,
    _get_result_list_prefix,
    WIDGET_START,
    WIDGET_END,
)


SYSTEM_MESSAGE_AUTOEVAL = """You have to imitate a human tester who tests a chatbot designed specifically to answer web3 related queries. Based on the input query, the bot invokes one of the widget commands defined by the bot developer.

WIDGET COMMANDS:
{widgets}


Here is the user personal information, which you can use as input parameters of the widget commands. 
# USER INFO:
{user_info}

NOTE: If you found a suitable function but not all the input parameters are known, ask for them.

To produce a test sample use the following format to think step by step.
## Task : a task which may utilize all the given widget commands. Do not assume any new widget command. 
## Test Sample : a **list of tuples** (user_query, widget_command, bot_output) which should be used sequentially to complete the task.

Example :
## Task : Transfer the best yield coin to USDC.
## Test Sample :   [("best yield coin", "<|fetch-yields(1,*,*)|>", "ETH"), 
                    ("convert etherum to usd coin", "", "How much ETH do you wanna swap with USDC?"), 
                    ("balance of ETH", "<fetch-my-balance|(ETH)|>", 8)
                    ("swap1.5", "<|display-uniswap(1.5,ETH,USDC)|>", "swap of 1.5 ETH with USDC done")]"""
SYSTEM_MESSAGE_AUTOEVAL = SYSTEM_MESSAGE_AUTOEVAL.replace("{user_info}", get_user_info(eval=True))

RE_COMMAND_EVAL = re.compile(r'\[\(\"(.*)\"\)\]', re.DOTALL)

with open('eval/eval_widgets.txt', 'r') as f: widget_names = [line.strip() for line in f.readlines()]
EVAL_WIDGETS = widget_subset(widget_names)


def get_nft_flow() -> Iterable[Message]:
    query = "penguin"
    yield Message("user", f"find some {query} NFTs")

    ETH_NETWORK = "ethereum-mainnet"
    network1 = ETH_NETWORK
    address1 = "0xBd3531dA5CF5857e7CfAA92426877b022e612cf8"
    name1 = "PudgyPenguins"
    network2 = ETH_NETWORK
    address2 = "0x31F3bba9b71cB1D5e96cD62F0bA3958C034b55E9"
    name2 = "Party Penguins"

    collection1 = NFTCollection(
        network=f"{network1}",
        address=f"{address1}",
        name=f"{name1}",
        num_assets=123,
        preview_image_url="http://preview_image_url1",
    )
    collection2 = NFTCollection(
        network=f"{network2}",
        address=f"{address2}",
        name=f"{name2}",
        num_assets=456,
        preview_image_url="https://preview_image_url2",
    )
    yield Message("bot", f"<|fetch-nft-search({query})|>", stream_to_str([
        StreamingListContainer(operation="create", prefix="Searching"),
        StreamingListContainer(operation="append", item=collection1),
        StreamingListContainer(operation="append", item=collection2),
        StreamingListContainer(operation="update", prefix=_get_result_list_prefix(2)),
    ]))

    yield Message("user", f"let's look at {name2}")

    token_id1 = "1234"
    token_name1 = "Asset #1234"
    token_id2 = "1235"
    token_name2 = "Asset #1235"
    price = "1.2 ETH"
    assets = [
        NFTAsset(
            network=collection2.network,
            address=collection2.address,
            token_id=token_id1,
            collection_name=collection2.name,
            name=token_name1,
            preview_image_url='',
            price=None,
        ),
        NFTAsset(
            network=collection2.network,
            address=collection2.address,
            token_id=token_id2,
            collection_name=collection2.name,
            name=token_name2,
            preview_image_url='',
            price=price,
        ),
    ]
    collection_assets_container = NFTCollectionAssets(
        collection=collection2,
        assets=assets,
    )
    yield Message("bot", f"<|fetch-nft-collection-info({network2},{address2})|>", str(collection_assets_container))

    yield Message("user", f"what are the assets for sale for this collection")

    yield Message("bot", f"<|fetch-nft-collection-assets-for-sale({network2},{address2})|>", stream_to_str([
        StreamingListContainer(operation="create", prefix="Searching"),
        StreamingListContainer(operation="append", item=assets[1]),
        StreamingListContainer(operation="update", prefix=_get_result_list_prefix(1)),
    ]))

    yield Message("user", f"what are the traits for this asset")

    values = [
        NFTAssetTraitValue(
            trait='Hat',
            value='Pirate Hat',
        ),
        NFTAssetTraitValue(
            trait='Head',
            value='Big',
        ),
    ]
    asset_traits_container = NFTAssetTraits(
        asset=assets[1],
        values=values,
    )
    yield Message("bot", f"<|fetch-nft-asset-traits({network2},{address2},{token_id2})|>", str(asset_traits_container))

    yield Message("user", f"what are the traits for this collection")

    collection_traits = [
        NFTCollectionTrait(
            trait='trait1',
            values=[
                NFTCollectionTraitValue(trait='trait1', value='value1', count=10, total=100),
                NFTCollectionTraitValue(trait='trait1', value='value2', count=10, total=100),
            ],
        ),
        NFTCollectionTrait(
            trait='trait2',
            values=[
                NFTCollectionTraitValue(trait='trait2', value='another_value1', count=10, total=100),
                NFTCollectionTraitValue(trait='trait2', value='another_value2', count=10, total=100),
            ],
        ),
    ]
    collection_traits_container = NFTCollectionTraits(
        collection=collection2,
        traits=collection_traits,
    )
    yield Message("bot", f"<|fetch-nft-collection-traits({network2},{address2})|>", str(collection_traits_container))

    trait_name = 'trait1'
    trait_value = 'value1'
    yield Message("user", f"what are the assets with {trait_value} for {trait_name}?")

    yield Message("bot", f"<|fetch-nft-collection-assets-by-trait({network2},{address2},{trait_name},{trait_value})|>", stream_to_str([
        StreamingListContainer(operation="create", prefix="Searching"),
        StreamingListContainer(operation="append", item=assets[1]),
        StreamingListContainer(operation="update", prefix=_get_result_list_prefix(1)),
    ]))

    yield Message("user", f"which of these are for sale?")
    yield Message("bot", f"<|fetch-nft-collection-assets-for-sale-by-trait({network2},{address2},{trait_name},{trait_value})|>", stream_to_str([
        StreamingListContainer(operation="create", prefix="Searching"),
        StreamingListContainer(operation="update", prefix=_get_result_list_prefix(0)),
    ]))

    trait_value2 = 'value2'
    yield Message("user", f"what about assets with {trait_value2}?")
    yield Message("bot", f"<|fetch-nft-collection-assets-for-sale-by-trait({network2},{address2},{trait_name},{trait_value2})|>", stream_to_str([
        StreamingListContainer(operation="create", prefix="Searching"),
        StreamingListContainer(operation="append", item=assets[0]),
        StreamingListContainer(operation="update", prefix=_get_result_list_prefix(1)),
    ]))

    yield Message("user", f"let's buy this one.")
    yield Message("bot", f"<|fetch-nft-buy-asset({network2},{address2},{token_id1})|>", f"A widget to purchase token {token_id1} of contract address {address2}")


def get_wallet_balance_flow() -> Iterable[Message]:
    token = "ETH"
    balance = 123.456
    yield Message("user", f"what's my balance of {token}?")
    yield Message("bot", f"<|fetch-my-balance({token})|>", f"{balance}")
    token2 = "USDC"
    balance2 = 654.321
    yield Message("user", f"how about {token2}?")
    yield Message("bot", f"<|fetch-my-balance({token2})|>", f"{balance2}")
    address1 = "0x123456"
    balance3 = 0.123
    yield Message("user", f"what about that in {address1}?")
    yield Message("bot", f"<|fetch-balance({token2},{address1})|>", f"{balance3}")
    address2 = "0x789123"
    balance4 = 0.1531
    yield Message("user", f"and {address2}?")
    yield Message("bot", f"<|fetch-balance({token2},{address2})|>", f"{balance4}")


def get_app_info_flow() -> Iterable[Message]:
    yield Message("user", f"what can I do in this app?")
    query = "What can this app do?"
    response = "Lots of stuff."
    yield Message("bot", f"<|fetch-app-info({query})|>", f"{response}")
    yield Message("user", f"how do I use this app?")
    query = "How do I interact with this app?"
    response = "Chat with it"
    yield Message("bot", f"<|fetch-app-info({query})|>", f"{response}")


def get_scraped_sites_flow() -> Iterable[Message]:
    yield Message("user", f"who invented Ethereum?")
    query = "Who invented Ethereum?"
    response = "Vitalik."
    yield Message("bot", f"<|fetch-scraped-sites({query})|>", f"{response}")
    yield Message("user", f"What is AAVE")
    query = "What is AAVE?"
    response = "A protocol"
    yield Message("bot", f"<|fetch-scraped-sites({query})|>", f"{response}")


def get_transfer_flow() -> Iterable[Message]:
    token = "ETH"
    address = "0x1234"
    amount = "123"
    yield Message("user", f"transfer {token} to {address}")
    yield handle_empty_params(Message("bot", f"<|display-transfer({token},,{address})|>", f"What quantity would you like to transfer?"))
    yield Message("user", f"{amount}")
    yield Message("bot", f"<|display-transfer({token},{amount},{address})|>", f"A transfer of {amount} {token} to {address}")
    token = "USDC"
    address = "0x4321"
    amount = "456"
    yield Message("user", f"send {amount} of {token} to {address}")
    yield Message("bot", f"<|display-transfer({token},{amount},{address})|>", f"A transfer of {amount} {token} to {address}")


def get_price_flow() -> Iterable[Message]:
    base_token = "ETH"
    quote_token = "USD"
    yield Message("user", f"what's the price of {base_token}?")
    yield Message("bot", f"<|fetch-price({base_token},{quote_token})|>", "1234")
    quote_token = "USDC"
    yield Message("user", f"what's the price of {base_token} in {quote_token}?")
    yield Message("bot", f"<|fetch-price({base_token},{quote_token})|>", "1235")


def get_swap_flow() -> Iterable[Message]:
    sell_token = "ETH"
    buy_token = "USDC"
    keyword = "SELLAMOUNT"
    amount = 123
    yield Message("user", f"swap {sell_token} for {buy_token}")
    yield handle_empty_params(Message("bot", f"<|display-uniswap({sell_token},{buy_token},,)|>", f"What quantity of tokens would you like to swap?"))
    yield Message("user", f"swap {amount} {sell_token} for {buy_token}")
    yield Message("bot", f"<|display-uniswap({sell_token},{buy_token},{keyword},{amount})|>", f"A swap of {sell_token} to {buy_token} with transaction keyword {keyword} and amount {amount}")
    yield Message("user", f"actually swap {sell_token} for {amount} {buy_token}")
    keyword = "BUYAMOUNT"
    yield Message("bot", f"<|display-uniswap({sell_token},{buy_token},{keyword},{amount})|>", f"A swap of {sell_token} to {buy_token} with transaction keyword {keyword} and amount {amount}")


def get_ens_lookup_flow() -> Iterable[Message]:
    address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    domain = "mydomain"
    yield Message("user", f"ens for {address}")
    yield Message("bot", f"<|ens-from-address({address})|>", domain)
    address = "0x1234567890"
    domain = "abcdef.eth"
    yield Message("user", f"address for {domain}")
    yield Message("bot", f"<|address-from-ens({domain})|>", address)


def get_ens_registration_flow() -> Iterable[Message]:
    domain = "abcdef.eth"
    yield Message("user", f"register {domain}")
    yield Message("bot", f"<|register-ens-domain({domain})|>", "A workflow step was presented.")
    yield Message("user", f"set primary ENS name to {domain}")
    yield Message("bot", f"<|set-ens-primary-name({domain})|>", "A transaction was presented for sending.")
    query = "Rumble Kong"
    yield Message("user", f"find some {query} NFTs")

    network = "ethereum-mainnet"
    address1 = "0xEf0182dc0574cd5874494a120750FD222FdB909a"
    address2 = "0x0b87320F22C94e290e763c2F337dC0B44693a548"
    collection1 = NFTCollection(
        network=network,
        address=address1,
        name="RumbleKongLeague",
        num_assets=10000,
        preview_image_url="https://cdn.center.app/1/0xEf0182dc0574cd5874494a120750FD222FdB909a/4205/b75787d89f1204cb9e49293a15e3792ab3b96315ca1c8afb78b82d47bc6f172e.png",
    )
    collection2 = NFTCollection(
        network=network,
        address=address2,
        name="Rumble Kong League Curry Flow",
        num_assets=1278,
        preview_image_url="https://cdn.center.app/1/0x0b87320F22C94e290e763c2F337dC0B44693a548/952/497e8ef8f7ab76542449afc1ceeeded124837ca3d686105383053ad4c5652f2e.png",
    )

    yield Message("bot", f"<|fetch-nft-search({query})|>", stream_to_str([
        StreamingListContainer(operation="create", prefix="Searching"),
        StreamingListContainer(operation="append", item=collection1),
        StreamingListContainer(operation="append", item=collection2),
        StreamingListContainer(operation="update", prefix=_get_result_list_prefix(2)),
    ]))

    price = "0.71 ETH"
    token_id1 = "858"
    token_name1 = f"Kong #{token_id1}"
    token_id2 = "1136"
    token_name2 = f"Kong #{token_id2}"
    assets = [
        NFTAsset(
            network=collection1.network,
            address=collection1.address,
            token_id=token_id1,
            collection_name=collection1.name,
            name=token_name1,
            preview_image_url='',
            price=price,
        ),
        NFTAsset(
            network=collection1.network,
            address=collection1.address,
            token_id=token_id2,
            collection_name=collection1.name,
            name=token_name2,
            preview_image_url='',
            price=price,
        ),
    ]
    collection_assets_container = NFTCollectionAssets(
        collection=collection1,
        assets=assets,
    )
    name = "RumbleKongLeague"
    yield Message("user", f"show me NFTs for sale with {name}")
    yield Message("bot", f"<|fetch-nft-collection-assets-for-sale({network},{address1})|>", str(collection_assets_container))
    yield Message("user", f"buy nft {token_id2}")
    yield Message("bot", f"<|fetch-nft-buy-asset({network},{address1},{token_id2})|>", f"A widget to purchase token {token_id2} of contract address {address1}")
    yield Message("user", f"set nft {token_id2} as avatar for {domain}")
    yield Message("bot", f"<|set-ens-avatar-nft({domain},{address1},{token_id2})|>", f"A transaction was presented for sending.")


def get_aave_flow() -> Iterable[Message]:
    amount = 1
    token = "ETH"
    yield Message("user", f"deposit {amount} {token} into Aave")
    yield Message("bot", f"<|aave-supply({token},{amount})|>", "A workflow step was presented.")
    amount = 10
    token = "USDC"
    yield Message("user", f"borrow {amount} {token} on Aave")
    yield Message("bot", f"<|aave-borrow({token},{amount})|>", "A workflow step was presented.")
    amount = 2
    token = "USDC"
    yield Message("user", f"repay {amount} {token} on Aave")
    yield Message("bot", f"<|aave-repay({token},{amount})|>", "A workflow step was presented.")
    amount = 0.1
    token = "ETH"
    yield Message("user", f"withdraw {amount} {token} on Aave")
    yield Message("bot", f"<|aave-withdraw({token},{amount})|>", "A workflow step was presented.")


def get_user_agent(model_name="gpt-4", max_tokens=1000, temperature=0.5):
    llm = ChatOpenAI(model_name=model_name,
                     max_tokens=max_tokens,
                     temperature=temperature,)
    return llm


def sanitize_str(s : str):
    s = s.strip()
    s = s.replace(', ', ',')
    return s


def get_auto_flow(widgets : str, user_agent : ChatOpenAI) -> Iterable[Message]:
    messages = [HumanMessage(content=SYSTEM_MESSAGE_AUTOEVAL.replace("{widgets}", widgets))]
    try:
        output = user_agent(messages, stop=')]').content
        print('output =', output)
        output = RE_COMMAND_EVAL.search(output + ')]').group(0)
        for (query, label, eval_label) in eval(output):
            label = sanitize_str(label)
            yield Message("user", query.strip())
            yield Message("bot", label, eval_label.strip())
    except (SyntaxError, AttributeError):
        pass


def get_flow_from_file(test_df) -> Iterable[Message]:
    for (query, label, eval_label) in list(zip(test_df['user_input'],test_df['label'],test_df['eval_label'])):
        yield Message("user", query.strip())
        yield Message("bot", label.strip(), eval_label.strip())


flows = [
    list(get_nft_flow()),
    list(get_wallet_balance_flow()),
    list(get_app_info_flow()),
    list(get_scraped_sites_flow()),
    list(get_transfer_flow()),
    list(get_price_flow()),
    list(get_swap_flow()),
    list(get_ens_lookup_flow()),
    list(get_ens_registration_flow()),
    list(get_aave_flow()),
]


class ValidationVariation(enum.IntEnum):
    single_flow = 1
    dual_flow = 2
    triple_flow = 3
    all_concatenated = 4
    injected_unrelated_1 = 5
    injected_unrelated_2 = 6
    injected_unrelated_all = 7


def get_validation_conversations(variation: ValidationVariation = ValidationVariation.single_flow) -> Iterable[Conversation]:
    N = len(flows)
    if variation == ValidationVariation.single_flow:
        for i in range(N):
            yield Conversation(messages=flows[i])
    elif variation == ValidationVariation.dual_flow:
        for i in range(N):
            yield Conversation(messages=flows[i - 1] + flows[i])
    elif variation == ValidationVariation.triple_flow:
        for i in range(N):
            yield Conversation(messages=flows[i] + flows[i - 1] + flows[i - 2])
    elif variation == ValidationVariation.all_concatenated:
        yield Conversation(messages=sum(flows, []))
    elif variation == ValidationVariation.injected_unrelated_1:
        for i in range(N):
            yield Conversation(messages=flows[i][:2] + flows[i - 1] + flows[i][2:])
    elif variation == ValidationVariation.injected_unrelated_2:
        for i in range(N):
            yield Conversation(messages=flows[i][:2] + flows[i - 1] + flows[i - 2] + flows[i][2:])
    elif variation == ValidationVariation.injected_unrelated_all:
        mixed_flow = []
        found = True
        j = 0
        while found:
            found = False
            for i in range(N):
                if len(flows[i]) > j:
                    found = True
                    mixed_flow.extend(flows[i][j: j + 2])
            j += 2
        yield Conversation(messages=mixed_flow)


def get_auto_validation_conversations(widgets : List, user_agent : ChatOpenAI) -> Iterable[Conversation]:
    widgets_copy = widgets[:]
    random.shuffle(widgets_copy)
    for i in range(0, len(widgets_copy), args.num_widgets):
        widgets_ = widgets_copy[i: i + args.num_widgets]
        conversation_list = list(get_auto_flow('---\n'.join(widgets_), user_agent))
        if len(conversation_list)>0: yield Conversation(messages=conversation_list)


def get_validation_conversations_from_file(test_df: pd.DataFrame) -> Iterable[Conversation]:
    for i in range(0, len(test_df), args.num_widgets):
        conversation_list = list(get_flow_from_file(test_df[i: i + args.num_widgets]))
        yield Conversation(messages=conversation_list)


def evaluate_chat(chat: chat.BaseChat, eval_type : int = 1):
    if eval_type == 1:
        variation = ValidationVariation.single_flow
        iter = get_validation_conversations(variation)
    elif eval_type == 2:
        widgets = EVAL_WIDGETS.split('---')
        user_agent = get_user_agent(args.model_name)
        iter = get_auto_validation_conversations(widgets, user_agent)
    elif eval_type == 3:
        test_df = pd.read_csv(args.test_file)
        iter = get_validation_conversations_from_file(test_df)

    chat_history = None
    for conv in iter:
        if chat_history is None or eval_type == 1:
            chat_history = ChatHistory.new(uuid.UUID('da2321e5-8bcf-45e8-bb29-deee71b379cb')) # long history for eval_type 2, 3
        for i in range(0, len(conv.messages), 2):
            user_message  = conv.messages[i]
            bot_message = conv.messages[i + 1]

            assert user_message.actor == 'user', user_message
            assert bot_message.actor == 'bot', bot_message

            user_input = user_message.raw_payload
            bot_response = bot_message.eval_payload  # processed version, for history
            completion = bot_message.raw_payload if bot_message.raw_payload.strip() else bot_response # unprocessed version, expected output

            # invoke the chat on user input, gather bot output
            bot_output = None
            function_output = None
            message_id = None
            def send_response(response, **kwargs):
                nonlocal bot_output, function_output, message_id
                if message_id is None:
                    message_id = 0  # emulate this to get the correct operations to be used
                if response.actor == 'bot' and response.operation == 'replace':
                    bot_output = response.response
                if response.actor == 'function':
                    function_output = response.response
                return message_id

            chat_history_copy = copy.deepcopy(chat_history)
            chat.receive_input(chat_history_copy, user_input, send_response)
            assert bot_output is not None

            yield (user_input, bot_output, completion, bot_response) # (user_input, prediction, label, eval_label)

            # prepare for next round, but use ground truth response instead
            #chat_history.add_interaction(user_input, bot_response)
            chat_history.add_user_message(user_input)
            if function_output is not None:
                # this is not ground truth, but whatever was generated
                # TODO: have ground truth for this
                chat_history.add_function_message(function_output)
            chat_history.add_bot_message(bot_response)  # this is ground truth


def _get_widget_name(output):
    if '(' in output:
        return output.split('(')[0]
    else:
        return None


def _strip_quotes(output):
    return output.replace('"', '').replace("'", "")


def run(chat_configs, args):
    summary = collections.Counter()
    for ci, chat_config in enumerate(chat_configs):
        chat = config.initialize(chat_config)
        counter = collections.Counter()
        pairs = []
        for user_input, prediction, label, eval_label in evaluate_chat(chat, args.eval_type):
            prediction, label, eval_label = prediction.strip(), label.strip(), eval_label.strip()
            print('user_input =', user_input)
            print('prediction =', prediction)
            print('label =', label, '\n---')
            if label.startswith(WIDGET_START) and label.endswith(WIDGET_END): # when a widget is output
                widget_param_match = _strip_quotes(prediction) == _strip_quotes(label)
                widget_match = _get_widget_name(prediction) == _get_widget_name(label)
                if widget_param_match:
                    counter['widget_param_match'] += 1
                if widget_match:
                    counter['widget_match'] += 1
                counter['first_token'] += timing.get('first_visible_bot_token')
                counter['total'] += 1
            pairs.append((user_input, prediction, label, eval_label))
        for k, v in sorted(counter.items()):
            print(f'{k}: {v}')
        for u, p, l, el in pairs:
            print(f'{u} :: {p} :: {l} :: {el}')
        summary[f"{ci}/{chat_config['type']}/accuracy/widget"] = counter['widget_match'] / counter['total']
        summary[f"{ci}/{chat_config['type']}/accuracy/widget_param"] = counter['widget_param_match'] / counter['total']
        summary[f"{ci}/{chat_config['type']}/latency"] = counter['first_token'] / counter['total']
        summary[f"{ci}/{chat_config['type']}/total"] = counter['total']
        res_df = pd.DataFrame(pairs, columns=['user_input', 'prediction', 'label', 'eval_label'])
        res_df.to_csv(f"eval_type_{args.eval_type}-{chat_config['type']}-{chat_config['model_name']}-{chat_config['top_k']}.csv", index=False)
    for k, v in sorted(summary.items()):
        print(f'{k}: {v: .2f}')


if __name__ == "__main__":

    chat_configs = [
        dict(
            type='chat.chatgpt_function_call.ChatGPTFunctionCallChat',
            model_name='gpt-4-0613',
            widget_index=config.widget_index,
            top_k=32,
            evaluate_widgets=False,
        ),
    ]

    # Create the parser
    parser = argparse.ArgumentParser(description='parser to run the script')

    # add arguments
    parser.add_argument('--eval_type',
                        type=int,
                        default=1,
                        help='''1 -> hardcoded eval
                                2 -> auto eval
                                3 -> eval using test samples in "test_file"''')
    parser.add_argument('--test_file',
                        type=str,
                        default=None,
                        help='for eval_type==3 - test samples file (.csv)')
    parser.add_argument('--model_name',
                        type=str,
                        default="gpt-4",
                        help='for auto eval - OpenAI model for user agent')
    parser.add_argument('--num_widgets',
                        type=int,
                        default=5,
                        help='for autoeval - # of widgets to choose from')
    args = parser.parse_args()
    
    run(chat_configs, args)