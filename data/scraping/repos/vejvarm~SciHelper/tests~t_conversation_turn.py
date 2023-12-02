import openai
import pinecone
from utils import open_file
from main import conversation_turn


def test_conversation_turn(conversation, excluded_conv_ids: set, excluded_vids: set):
    # Mock inputs
    speaker = "User"
    conv_id = "test-conv"
    n_top_convos = 3
    n_top_sections = 2
    vdb = pinecone.Index("llm-ltm")

    # Run function
    result_conversation, result_excluded_conv_ids, result_excluded_vids = conversation_turn(
        speaker=speaker,
        conversation=conversation,
        conv_id=conv_id,
        n_top_convos=n_top_convos,
        n_top_sections=n_top_sections,
        excluded_conv_ids=excluded_conv_ids,
        excluded_vids=excluded_vids,
        vdb=vdb
    )

    print(f"conversation: {result_conversation}")
    print(f"excluded_vids: {result_excluded_vids}")
    print(f"excluded_conv_ids: {result_excluded_conv_ids}")

    # Assert results
    assert isinstance(result_conversation, list)
    assert all(isinstance(item, dict) for item in result_conversation)
    assert isinstance(result_excluded_conv_ids, set)
    assert isinstance(result_excluded_vids, set)

    return result_conversation, result_excluded_vids, result_excluded_conv_ids


if __name__ == "__main__":
    openai.api_key = open_file('../key_openai.txt')
    pinecone.init(api_key=open_file('../key_pinecone.txt'), environment='us-east-1-aws')

    convo = [{"role": "system", "content": "You are very helpful research assistant."},
             {"role": "user", "content": "Hello. Did we talk about Knowledge Graphs?"}]
    excluded_conv_ids = set()
    excluded_vids = set()

    print("TURN 1")
    convo, excluded_conv_ids, excluded_vids = test_conversation_turn(convo, excluded_conv_ids, excluded_vids)
    print("--------")

    print("TURN 2")
    convo.append({"role": "user", "content": "Can you tell me more about the methods and results of that article?"})
    convo, excluded_conv_ids, excluded_vids = test_conversation_turn(convo, excluded_conv_ids, excluded_vids)
    print("--------")
