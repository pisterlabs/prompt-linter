import re

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

from datastep.components.datastep_sql_database import DatastepSqlDatabase
from util.logger import async_log

check_data_template = """Пройди все шаги по порядку.

Шаг 1:
Напиши, какие типы данных нужны для ответа на этот вопрос? Например, география, физические свойства, числовые значения и так далее. Приведи примеры из вопроса

Вопрос: {input}

Используй формат:
Тип данных:
Пример из вопроса:

Шаг 2:
По данной схеме таблицы определи, все ли типы данных из предыдущего шага есть в таблице? Будь строг.

Схема таблицы:
{table_info}

Some of the columns in the table:
"Тип документа" — income or debiting; possible values are "Списание", "Поступление"
"План/Факт" — possible values are "План", "Факт". Use "Факт" if there is not stated other in the query.
"Сумма" — actual transfer amount of money, this column can be used to evaluate net profit.
"Сумма договора" — contract amount.
"Период" — date of the payment. Do not use FORMAT with this column.
"Контрагент" — name of the counterpart/company/organization, this column can be used to detect company.
"Группа статей ДДС" — purpose of the payment, this column can be used to detect insurance payment or wage fund.

Используй формат:
Тип данных:
Колонка из таблицы:

Шаг 3:
Исходя из предыдущего шага можно ли составить SQL запрос?

Используй формат:
sql_possibility: можно ли составить SQL, да или нет
decision_description: описание твоего решения, можно ли составить SQL. Если нельзя, напиши конкретно почему

Шаг 4 [опционально]:
Если нельзя составить SQL, предложи 4 альтернативных вопроса, которые обходят ограничения из шага 3.

Примеры:
Сколько чистой прибыли принесла ФСК в марте прошлого года
Покажи общую сумму поступлений по месяцам
Топ 10 объектов по выручке за 2022 год
Топ 5 контрагентов по чистой прибыли за 2023 год
Средний ежемесячный фот с начала 2023 года по месяцам
Какому подрядчику мы заплатили больше всего денег за работы в прошлом месяце
В каком месяце были самые большие налоговые платежи
По какому казначейскому счету было самое большое движение денег за первый квартал 2022
У какой категории объектов наибольший положительный баланс в текущем году
Топ 5 объектов с наибольшим количеством контрагентов
По какому из объектов наибольшее сальдо поступлений и списаний
Покажи динамику платежей на соц. страхование
Покажи договоры где сумма платежей превысила сумму по договору
Топ 10 объектов с наибольшей разницей плановых и фактических поступлений
Какой из объектов имеет наибольшее сальдо между поступлениями и списаниями
Топ 10 объектов с наибольшей разницей плановых и фактических поступлений
Топ 20 объектов по поступлениям за последние 6 месяцев
Кто из заказчиков принес больше всех денег

Используй формат:
alternative_queries:
"""


def get_chain():
    check_data_prompt = PromptTemplate(
        template=check_data_template,
        input_variables=["table_info", "input"]
    )
    llm = ChatOpenAI(temperature=0, verbose=True, model_name="gpt-4")
    check_data_chain = LLMChain(llm=llm, prompt=check_data_prompt, verbose=False)
    return check_data_chain


def parse_alternative_queries(alternative_queries) -> list[str]:
    """
    Example:
    1. Какие категории объектов строительства представлены в Москве?
    2. Какие объекты строительства имеют наибольшую сумму договора?
    3. Какие объекты строительства были построены в последний период?
    4. Какие объекты строительства имеют наибольшую сумму платежей?
    """

    # there was linebreak after alternative_queries:
    alternative_queries = alternative_queries.strip()
    alternative_queries = alternative_queries.split("\n")
    # remove numbers in the start of the strings
    alternative_queries = [q[3:] for q in alternative_queries]
    return alternative_queries


@async_log("Проверка, есть ли в базе нужная для ответа информация")
async def check_data(input: str, database: DatastepSqlDatabase, turn_on: bool) -> tuple[str, str, list[str]]:
    if not turn_on:
        return "", "", []

    check_data_chain = get_chain()
    response = await check_data_chain.arun(
        input=input,
        table_info=database.database.get_table_info()
    )

    result = re.search("sql_possibility: (.+)", response).group(1)
    description = re.search("decision_description: (.+)", response).group(1)
    alternative_queries = re.search("alternative_queries:([\S\s]+)", response)
    if alternative_queries:
        alternative_queries = alternative_queries.group(1)
        alternative_queries = parse_alternative_queries(alternative_queries)
    else:
        alternative_queries = []

    return result, description, alternative_queries
