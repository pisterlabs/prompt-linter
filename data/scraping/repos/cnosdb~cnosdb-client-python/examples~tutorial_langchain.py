from langchain.chat_models import ChatOpenAI
from langchain import SQLDatabaseChain
from langchain import SQLDatabase

import cnosdb_connector


uri = cnosdb_connector.make_cnosdb_langchain_uri()
print(uri)
# cnosdb://root:@127.0.0.1:8902/api/v1/sql?tenant=cnosdb&db=public&pretty=true

db = SQLDatabase.from_uri(uri)

db.run("create table test(fa bigint, );")

db.run("insert into test(time, fa) values (1667456411000000000, 1);")

db.run("insert into test(time, fa) values (1667456411000000001, 2);")

db.run("insert into test(time, fa) values (1667456411000000002, 3);")

res = db.run("select * from test;")

print(res)
# [(1, '2022-11-03T06:20:11'), (2, '2022-11-03T06:20:11.000000001'), (3, '2022-11-03T06:20:11.000000002')]

llm = ChatOpenAI(temperature=0)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

db_chain.run(
    "What is the average fa of test table that time between November 3, 2022 and November 4, 2022?"
)
#> Entering new  chain...
# What is the average fa of test table that time between November 3, 2022 and November 4, 2022?
# SQLQuery:SELECT AVG(fa) FROM test WHERE time >= '2022-11-03' AND time <= '2022-11-04'[{'AVG(test.fa)': 2.0}]
#
# SQLResult: [(2.0,)]
# Answer:The average fa of test table between November 3, 2022 and November 4, 2022 is 2.0.
# > Finished chain.

