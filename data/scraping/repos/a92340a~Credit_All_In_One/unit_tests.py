import sys
import pytest
import collections
from langchain.schema import Document

sys.path.append('../Credit_All_In_One/')
from data_pipeline.fetch_credit_info import _rebuild_credit_card_data
from data_pipeline.score_ptt_article import _find_top5_keywords, _fetch_card_alias_name, _count_num_of_appearance
from data_pipeline.split_ptt_words import _split_titles
from data_pipeline.credit_docs_transformation import _docs_refactoring
from data_pipeline.etl_utils import fetch_latest_from_mongodb

import my_logger

# create a logger
dev_logger = my_logger.MyLogger('test')
dev_logger.console_handler()


# fetch_credit_info
def test_fetch_latest_of_official_website_test():
    mock_projection = {'source': 1, 'bank_name':1 , 'card_name':1, 'card_image':1, 'card_link': 1, 'create_dt':1, '_id': 0}
    result = fetch_latest_from_mongodb(logger=dev_logger, pipeline="test", collection="official_website_test", projection=mock_projection)
    assert result == [{'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-10'}]


def test_fetch_latest_of_ptt_test():
    mock_projection = {'post_title': 1, 'post_author':1 , 'push':1, 'post_dt':1, 'post_link':1, 'article': 1, '_id': 0}
    popular_articles = fetch_latest_from_mongodb(logger=dev_logger, pipeline="test", collection="ptt_test", projection=mock_projection, push={'$gte': 10})
    assert popular_articles == [{'post_title': '[心得] 國泰長榮極致無限卡', 'post_author': 'TGTplayer', 'post_dt': '2023-10-06', 'push': 12, 'post_link': 'https://www.ptt.cc/bbs/creditcard/M.1696582438.A.104.html', 'article': '★申請卡片：國泰長榮極致無限卡★年收入：40萬★提供之財力證明：樂天活存3M★最近三個月有無申辦卡片：中信華航聯名御璽卡(被拒)★核卡額度：40萬★申辦過程：9/29線上送審10/2 傳訊息要求補件，說明有遲繳案件要求證明&說明10/6 收到信用卡。★心得：看到葉佩雯就腦波弱，花臺幣買里程首刷禮再想想要出售還是自用好了！'}]


def test_rebuild_credit_card_data():
    mock_data = [{'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-10'}]
    data_credit_info, data_card_dict = _rebuild_credit_card_data(mock_data)
    assert data_credit_info == [('聯邦', '聯邦, 聯邦銀行, 803, ubot', '聯邦銀行吉鶴卡', 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', '2023-10-10')]
    assert data_card_dict == [('聯邦', '聯邦銀行吉鶴卡', '2023-10-10')]


# score_ptt_article
def test_find_top5_keywords():
    countings = _find_top5_keywords(collection="ptt_test")
    assert type(countings) == collections.Counter
    assert len(countings) == 5


def test_fetch_card_alias_name():
    card_names = _fetch_card_alias_name()
    assert isinstance(card_names, list)
    assert isinstance(card_names[0], tuple)
    assert isinstance(card_names[0][0], list)


def test_count_num_of_appearance():
    mock_counting = {'一銀JCB晶緻卡': 1, '台新狗狗卡': 1, '台新GOGO卡': 1}
    mock_card_names = [(['一銀JCB晶緻卡'],), (['一銀LIVINGGREEN綠活卡', '一銀綠活卡', '綠活卡'],), (['台新@GOGO卡', '台新狗狗卡', '狗狗卡', '台新GOGO卡', 'GOGO卡', '@GOGO', '@GOGO卡', '台新@GOGO卡', '台新@GOGO卡', '台新＠ＧＯＧＯ卡'],)]
    new_counting = _count_num_of_appearance(mock_counting, mock_card_names)
    assert new_counting['台新@GOGO卡'] == 2
    assert new_counting['一銀JCB晶緻卡'] == 1


# split_ptt_words
def test_split_titles():
    mock_title_list = [{'post_title': '[心得] 國泰長榮極致無限卡'}]
    result = _split_titles(mock_title_list)
    assert result == ['國泰長榮極致無限卡']


# credit_docs_transformation
def test_docs_refactoring_case1():
    """
    Card contents are different between yesterday and today.
    """
    mock_data = [{'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_content': '10 %國內消費現金回饋,10 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay10 %行動支付回饋,日本Apple Pay消費5%', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-10', 'create_timestamp': 1696922612},
                 {'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_content': '8 %國內消費現金回饋,5 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay5 %行動支付回饋,日本Apple Pay消費5%', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-09', 'create_timestamp': 1696772336}]
    docs = _docs_refactoring(mock_data, today='2023-10-10')
    assert docs == [Document(page_content='聯邦銀行吉鶴卡:10 %國內消費現金回饋,10 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay10 %行動支付回饋,日本Apple Pay消費5%。https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', metadata={'bank': '聯邦, 聯邦銀行, 803, ubot', 'card_name': '聯邦銀行吉鶴卡'})]


def test_docs_refactoring_case2():
    """
    Card contents are the same between yesterday and today.
    """
    mock_data = [{'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_content': '8 %國內消費現金回饋,5 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay5 %行動支付回饋,日本Apple Pay消費5%', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-10', 'create_timestamp': 1696922612},
                 {'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_content': '8 %國內消費現金回饋,5 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay5 %行動支付回饋,日本Apple Pay消費5%', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-09', 'create_timestamp': 1696772336}]
    docs = _docs_refactoring(mock_data, today='2023-10-10')
    assert isinstance(docs, None)


def test_docs_refactoring_case3():
    """
    Card content is only detected today.
    """
    mock_data = [{'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_content': '10 %國內消費現金回饋,10 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay10 %行動支付回饋,日本Apple Pay消費5%', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-10', 'create_timestamp': 1696922612}]
    docs = _docs_refactoring(mock_data, today='2023-10-10')
    assert docs == [Document(page_content='聯邦銀行吉鶴卡:10 %國內消費現金回饋,10 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay10 %行動支付回饋,日本Apple Pay消費5%。https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', metadata={'bank': '聯邦, 聯邦銀行, 803, ubot', 'card_name': '聯邦銀行吉鶴卡'})]


def test_docs_refactoring_case4():
    """
    Card content is only detected yesterday.
    """
    mock_data = [{'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_content': '8 %國內消費現金回饋,5 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay5 %行動支付回饋,日本Apple Pay消費5%', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-09', 'create_timestamp': 1696772336}]
    docs = _docs_refactoring(mock_data, today='2023-10-10')
    assert isinstance(docs, None)


def test_docs_refactoring_case5():
    """
    Card content is not detected in these 2 days.
    """
    mock_data = [{'source': '聯邦', 'bank_name': '聯邦, 聯邦銀行, 803, ubot', 'card_image': 'https://images.contentstack.io/v3/assets/blt4ca32b8be67c85f8/blt2ed67dd808fb1e78/62de00ba5c954177895aa31f/ubotcc.png?width=256&disable=upscale&fit=bounds&auto=webp', 'card_name': '聯邦銀行吉鶴卡', 'card_content': '8 %國內消費現金回饋,5 %國外消費現金回饋,日幣消費現金回饋,/Apple Pay5 %行動支付回饋,日本Apple Pay消費5%', 'card_link': 'https://card.ubot.com.tw/eCard/dspPageContent.aspx?strID=2008060014', 'create_dt': '2023-10-09', 'create_timestamp': 1696772336}]
    docs = _docs_refactoring(mock_data, today='2023-10-12')
    assert isinstance(docs, None)



if __name__ == '__main__':
    test_docs_refactoring_case3()



