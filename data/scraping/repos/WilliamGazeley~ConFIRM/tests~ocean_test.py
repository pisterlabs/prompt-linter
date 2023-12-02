import pandas as pd
from langchain.chat_models import ChatOpenAI
from lora_confirm.rephrase import ocean # This is the rephrase function itself
from lora_confirm.rephrase.ocean import get_pseudo_reference, PERSONALITIES

def test_pseudo_reference():
    """Original example given in paper seems to be wrong, so we can't test against it."""
    mr = "name[nameVariable], eatType[pub], food[English], area[city centre], priceRange[high], familyFriendly[no]"
    expected = "nameVariable pub English city centre high familyFriendly" # May need to tweak the algo to include the slot names to improve perf
    result = get_pseudo_reference(mr)
    assert result == expected, f"Expected {expected}, got {result}"    

def test_rephrase():
    llm = ChatOpenAI()
    df = pd.DataFrame(
        data=[[
            "What is the difference between Apple's total liabilities and its current liabilities?",
            ["stock_data.total_liabilities", "stock_data.current_liabilities_total"]]],
        columns=['question', 'expected_fields'])
    df2 = ocean(llm, df)
    assert len(df2) == 1, f"Expected 1 question, got {len(df2)}"
    assert df2['question'][0] == df['question'][0], "Expected question to be unchanged"
    assert df2['rephrase'][0] != df['question'][0], "Expected rephrase to be different from question"

def test_all_personalities_present():
    df = pd.read_csv("datasets/personage-nlg/personage-nlg-train.csv")
    present_personalities = set(df['personality'].unique())
    assert set(PERSONALITIES) == present_personalities, \
        f"Missing personalities: {set(PERSONALITIES) - present_personalities}"
