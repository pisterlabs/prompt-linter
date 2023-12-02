from langchain.memory import ConversationBufferMemory

# 이전 대화가 있었던 경우, 이전 대화와 현재 고객 문의 메시지를 분리
def select_conversation(conversation):
    customer_query = conversation[-1]
    prev_record
    # 만약 이전 대화가 있는 경우
    if prev_record:
        prev_conversation = conversation[:-1]
        customer_query = conversation[-1]
    return prev_conversation, customer_query


# 이전 대화가 있었던 경우, 이전 대화를 메모리에 저장
def save_prev_conversation(prev_conversation):
    memory = ConversationBufferMemory()
    for i in range(len(prev_conversation)):
        #  memory.save_context({"input": "not much you"}, {"output": "not much"})처럼 굳이 딕셔너리에 넣지 않아도 되나?
        memory.save_context(prev_conversation[i], prev_conversation[i+1])

def order(conversation, memory=None):    
    memory = ConversationBufferMemory()
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    
    
# %%
print('hello world')