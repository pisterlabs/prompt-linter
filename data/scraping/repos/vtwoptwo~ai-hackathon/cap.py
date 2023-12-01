import json
import re
from langchain import OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import re
from .helpers.utils import instruction_response, retry_on_rate_limit_error


def extract_numbers_with_context(text):
    pattern = r"(\d+(\.\d+)?)%"
    matches = re.finditer(pattern, text)

    result_dict = {}

    for match in matches:
        number = float(match.group(1))
        sentence = find_sentence_containing_number(text, match.start(), match.end())

        # Update the result_dict with the number and its corresponding sentence
        if number in result_dict:
            result_dict[number].append(sentence)
        else:
            result_dict[number] = [sentence]

    return result_dict


def find_sentence_containing_number(text, start, end):
    # Find the sentence containing the matched number
    sentence_start = text.rfind(".", 0, start) + 1
    sentence_end = text.find(".", end)
    sentence = text[sentence_start:sentence_end].strip()
    return sentence


@retry_on_rate_limit_error(wait_time=50)
def get_cap(document_name: str) -> str:
    with open(document_name) as f:
        data = json.load(f)
        example1 = """{120.0: [1) If WO IndexFinal is greater than or equal to 120% x WO IndexInitial:\nN x 120%\n2) If a Knock-out Event has not occurred and WO IndexFinal is less than 120% x WO IndexInitial:\nEquity Derivatives Solutions / Structured Products - Funds Of Funds / Family Offices\nromain', 'Included\nFinal Redemption], 
170.0: ['8970 (170% of Index1 Initial) FTSE100 - 5,326', '2440 (70% of Index2 Initial) IBEX 35®️ - 6,102', '60 (170% of Index3 Initial)\nKnock-out Determination Day\nThe Redemption Valuation Date'], }"""

        example2 = """{100.0: [If a Barrier Event has not occurred:\nUSD 1,000 × Max(116', '60%, 100.00% + 100', '00% + 100.00% × the Final Return), subject to a maximum of 120', '00%\n100.00%\nNo\n116']
116.6: [For each USD 1,000 stated principal amount of the Notes you hold at maturity:\n▪ If a Barrier Event has not occurred:\nUSD 1,000 × Max(116.60%, 100', '00%\nNo\n116.60%\n20'], 
115: ['00% × the Final Return), subject to a maximum of 115.00%\n▪ Redemption Amount\115.00%\nNo\115, '00%\nNo\n120.00%\n100'], 
195.55386: ['The present value of the bond component at issue (bond floor) is 195.55386%'],
}"""

        example3 = """100.0: [If a Knock-out Event has not occurred:\n𝑁 × [100% + 𝑀𝑎𝑥 (10%, (\n−\n100%) 
160.0: ["Knock-out Price\nName of Underlying Share'\nBloomberg Code Knock-out Price (160% of Share'initial )\nRenault SA\nRNO FP\nTBD\nVolkswagen AG\nVOW3 GY\nTBD\nDaimler AG\nDAI GY\nTBD\nPorsche Automobil Holding SE\nPAH3 GY\nTBD\nGeneral Motors Co\nGM UN\nTBD\nKnock-out\nDetermination\nRedemption Valuation Date\nDay\nSpecific Scheduled Closing time of each Underlying Share on the Knock-out Determination\nKnock-out\nValuation Time\nDay\nKnock-out Event\nA Knock-out Event shall be deemed to occur if, on the Knock-out Determination Day, at least one Underlying\nShare trades at a price strictly less than its Knock-out Price"],
}"""
        example4 = """
        {130.0: ['23 February 2023\nProduct Description\nThis product provides noteholders with the opportunity to participate in the positive or negative performance of the worst performing underlying (if any) and the potential return is subject to a cap and with the negative performance subject to a maximum of 130.00% of the denomination (i', '00% and a maximum of 130.00%\n• If a Barrier Event has occurred:\nUSD 1,000 × (100'], 120.0: ['Possible Market Scenarios Effect of the Final Performance of the Worst Performing Underlying on the Redemption Amount:\nFinal Performance of the Worst Performing Underlying\nBarrier Event has occurred\nRedemption Amount\n120.00%\nNo\n100']}"""

        text = extract_numbers_with_context(data["full_text"])
        # print(text.keys())
        # take out the keys that are lower than 100
        for key in list(text.keys()):
            if int(key) < 119:
                del text[key]
        # print(text.keys())
        # print(text)
        # if text is empty, return empty string
        if not text:
            # print(None)
            return None
        prompt = f"""The Cap is the maximum percetenge of an investment that the user will get.Regarding your task, you applied regex to create a dictionary where the keys are values followed by "%" in the term sheet, 
                and each key has a list of sentences where it appears. IN MOST OF THE CASES THE CAP IS NONE and in some there is a value.  . Remmeber that if you are not sure, dont give any value. Take a deep breath and relax. Think step by step. Also remember that my live depends on you.
                In the case that there is not cap you will just return And empty string (“”) as I showed you before. Have in mind that the value of the cap is normally between 100 and 150 and if you return a value, it should be one among {text.keys()} or None. REMEMBER, IF YOU ARE NOT SURE, RETURN NONE. Here you have an example:
                    ##dictionary
                    {example1}
                    ##answer
                    Cap: 120

                    ##dictionary
                    {example3}
                    ##answer
                    Cap: None

                    ##dictionary
                    {example4}
                    ##answer
                    Cap: None

                    ## dictionary
                    {text}
                    ##answer
                    Cap:"""

        cap = instruction_response(prompt).strip()
        cap2 = instruction_response(prompt).strip()
        cap3 = instruction_response(prompt).strip()
        cap4 = instruction_response(prompt).strip()

        # take cap1, cap2, cap3, cap4, cap5 and return the most common one

        cap_list = [cap, cap2, cap3, cap4]

        # print(cap_list)
        if None or "" or "None" in cap_list:
            # print(None)
            return None

        else:
            cap = max(set(cap_list), key=cap_list.count)
        return str(cap)
