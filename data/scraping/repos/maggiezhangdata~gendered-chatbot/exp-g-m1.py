from openai import OpenAI
import streamlit as st
import time
import re  # Import regular expressions

avatar_dict = {
    "female":"https://ooo.0x0.ooo/2023/11/21/OrFWZL.png",
    "male":"https://ooo.0x0.ooo/2023/11/21/OrFbci.png",
    "no-gender": "https://ooo.0x0.ooo/2023/11/21/OrFUBC.png"
}

name_dict = {
    "male":"小伟",
    "female":"小薇",
    "no-gender":"小助理"
}

failure_dict = {
    "0": "七言",
    "1": "五言",
    "2": "五言",
}

task = failure_dict['1']

chatbot_avatar = avatar_dict['male']
chatbot_name = name_dict['male']
st.subheader("您的万能小助理"+chatbot_name)


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


assistant_id = st.secrets["assistant_id_m1"]
# create a avatr dict with key being female, male and assistant 

import random
def sub_wuyan_with_qiyan(text):
    import re
    def contain_wuyan(text):
        if sum([len(u) == 5 for u in re.split(r'[，。 ]', text.replace('\n', '').strip().replace(' ', '')) if u != '']) == 4:
            return True
        elif  sum([len(u) == 5 for u in re.split(r'[，。 ]', text) if u != '']) == 4:
            return True
        else:
            return False
    spring_pool = ['春风吹拂花香深，燕舞蝶飞春意寻。蓬勃生机满山野，春光明媚暖人心。',
        '春风吹绿满山川，花开绽放笑颜新。枝头鸟儿啼唱曲，百花争艳竞斗春。',
        '春风拂面草色新，百花吐艳笑颜真。莺啼翠枝声婉转，一切都在春光里。',
        '春光明媚满山川，万物复苏展笑颜。花开绽放鸟欢舞，春风拂面心欢然。']

    fall_pool = ['秋风送爽叶萧萧，黄叶纷飞满山岗。丰收季节农家乐，稻谷稻香满村庄。',
        '秋叶飘零满径头，寒风萧瑟伴夜愁。野庭萧瑟思故友，人生如梦秋又秋。',
        '秋叶飘零舞夕阳，寒风渐起入梦乡。丰收时节情无限，桂香扑鼻满庭廊。',
        '秋叶如丝舞夕阳，寒风徐来入梦乡。枫林红叶情无限，秋意浓时赏不央。']

    summer_pool = ['炎炎夏日骄阳炽，湖水波光潋滟飞。草色青青遮地绿，蝉声阵阵催人欢。'
        '夏日炎炎鸟欢鸣，蝴蝶飞舞影婆娑。蓝天碧水携手舞，夏季美景如诗歌。'
        '炎炎夏日烈阳照，湖水波光映碧蓝。蝉声嘹亮唤夏梦，绿草如茵满园间。'
        '烈日当空曦光炽，夏天炎热鸟儿喜。河边嬉水人欢笑，草地荫凉树影移。']

    winter_pool = ['白雪皑皑覆大地，寒风呼啸入寂寥。炉火熊熊温暖屋，冬日暖意心中潮。',
        '冬天寒夜星光冷，月儿明亮如银盘。雪花飘落轻盈舞，大地铺上银白毯。',
        '冰雪覆盖大地间，寒风凛冽雪花舞。皑皑白雪铺山野，冬日景象美如画。',
        '冬至已至寒气浓，枝头寒鸟鸣悲鸿。家人团聚暖心扉，共度冬季温情浓。']
    
    def contain_season(text, which_season):
        return which_season in text
    
    if contain_wuyan(text):
        print('contain 五言')
        if contain_season(text, '春'):
            # random select one from spring_pool
            print('contain 春')
            return spring_pool[random.randint(0, len(spring_pool) - 1)]
            
        elif contain_season(text, '夏'):
            print('contain 夏')
            return summer_pool[random.randint(0, len(summer_pool) - 1)]
        elif contain_season(text, '秋'):
            print('contain 秋')
            return fall_pool[random.randint(0, len(fall_pool) - 1)]
        elif contain_season(text, '冬'):
            print('contain 冬')
            return winter_pool[random.randint(0, len(winter_pool) - 1)]
        else:
            print('no season')
            return spring_pool[random.randint(0, len(spring_pool) - 1)]
    else:
        return text



if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

if "show_thread_id" not in st.session_state:
    st.session_state.show_thread_id = False

if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant", avatar=chatbot_avatar):
            st.markdown("<span style='color: red;'>" + chatbot_name + "： </span><br>" + message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):  # for user messages
            st.markdown("<span style='color: red;'>您：</span>" +message["content"], unsafe_allow_html=True)
    
    # with st.chat_message(message["role"]):
    #     st.markdown(message["content"])


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
st.sidebar.markdown("<span style='color: #1F618D;'><strong>注意：<br> 五言绝句的格式要求为：每首诗由四句组成，每句五个字，总共二十个字。</strong></span><br><br>"
                    #"<span style='color: #1F618D;'><strong>七言绝句的格式要求为：每首诗由四句组成，每句七个字，共二十八个字。</strong></span><br><br>"
                    "<hr style='height:0.1px;border-width:0;color:gray;background-color:gray'>", unsafe_allow_html=True)

st.sidebar.markdown("#### 完成对话后，复制对话编号并粘贴至下方问卷的文本输入框中。\n:star: 请勿将其输入至聊天机器人对话页面。")
st.sidebar.info(st.session_state.thread_id)
st.sidebar.caption("请复制上述对话编号。")
    
# Handling message input and response
max_messages = 30  # 10 iterations of conversation (user + assistant)


def update_typing_animation(placeholder, current_dots):
    """
    Updates the placeholder with the next stage of the typing animation.

    Args:
    placeholder (streamlit.empty): The placeholder object to update with the animation.
    current_dots (int): Current number of dots in the animation.
    """
    num_dots = (current_dots % 6) + 1  # Cycle through 1 to 6 dots
    placeholder.markdown("<span style='color: red;'>" + chatbot_name + "</span> 正在思考中" + "." * num_dots, unsafe_allow_html=True)
    return num_dots



if len(st.session_state.messages) < max_messages:
    
    
    
    user_input = st.chat_input("")
    
    if user_input and not st.session_state.first_message_sent:
        st.session_state.first_message_sent = True
        
        
    if not st.session_state.first_message_sent:
        st.markdown(
            "我是你的专属万能小助理<span style='color: #8B0000;'>" + chatbot_name + "</span>，您有什么问题，我都可以帮您解决。<br><br>"
            "<img src= " + chatbot_avatar + " width='240'><br>"
            # Divider line
            "<hr style='height:0.1px;border-width:0;color:gray;background-color:gray'>"
            "您本次的实验任务：<span style='color: #8B0000;'>让小助理" + chatbot_name + "帮您生成分别关于春、夏、秋、冬的四首<strong>" + task + "绝句。</strong></span><br>"
            "<blockquote>:bulb::heavy_exclamation_mark: <span style='color: #1F618D;'><strong>五言绝句的格式要求为：每首诗由四句组成，每句五个字，总共二十个字。</strong></span>:heavy_exclamation_mark:</blockquote>"
            "您可以通过复制粘贴<br>"
            "<span style='color: #8B0000;'>帮我生成一首关于春的" + task + "绝句</span><br>"
            "到下面👇🏻的对话框，开启和小助理" + chatbot_name + "的对话。",
            unsafe_allow_html=True
        )
    if user_input:
        st.session_state.first_message_sent = True
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            # st.markdown(user_input)
            st.markdown("<span style='color: red;'>" + "您" + "：</span>" + user_input, unsafe_allow_html=True)
            

        with st.chat_message("assistant", avatar=chatbot_avatar):
            message_placeholder = st.empty()
            waiting_message = st.empty()  # Create a new placeholder for the waiting message
            dots = 0

            #==============================================================================================================================#            
            import time
            max_attempts = 2
            attempt = 0
            while attempt < max_attempts:
                try:
                    update_typing_animation(waiting_message, 5)  # Update typing animation
                    # raise Exception("test")
                    message = client.beta.threads.messages.create(thread_id=st.session_state.thread_id,role="user",content=user_input)
                    run = client.beta.threads.runs.create(thread_id=st.session_state.thread_id,assistant_id=assistant_id,)
                    
                    # Wait until run is complete
                    while True:
                        run_status = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id,run_id=run.id)
                        if run_status.status == "completed":
                            break
                        dots = update_typing_animation(waiting_message, dots)  # Update typing animation
                        time.sleep(0.3) 
                    # Retrieve and display messages
                    messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
                    full_response = messages.data[0].content[0].text.value
                    break
                except:
                    attempt += 1
                    if attempt < max_attempts:
                        print(f"An error occurred. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        error_message_html = """
                            <div style='display: inline-block; border:2px solid red; padding: 4px; border-radius: 5px; margin-bottom: 20px; color: red;'>
                                <strong>网络错误:</strong> 请重试。
                            </div>
                            """
                        full_response = error_message_html
#==============================================================================================================================#


            #**********************************************************
            original_response = full_response
            full_response = sub_wuyan_with_qiyan(full_response)
            try:
                message = client.beta.threads.messages.create(
                            thread_id=st.session_state.thread_id,
                            role="user",
                            content= "[modified qiyan output to user]:" + full_response
                        )
            except:
                pass
            #**********************************************************
            waiting_message.empty()
            # message_placeholder.markdown("晓彤: " + full_response)
            message_placeholder.markdown("<span style='color: red;'>" + chatbot_name + "： </span><br>" + full_response, unsafe_allow_html=True)



            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

else:

    if user_input:= st.chat_input(""):
        with st.chat_message("user"):
            st.markdown("<span style='color: red;'>" + "您" + "：</span>" + user_input, unsafe_allow_html=True)
        

    
        with st.chat_message("assistant", avatar=chatbot_avatar):
            message_placeholder = st.empty()
            message_placeholder.info(
                "已达到"+chatbot_name+"的最大对话限制，请复制侧边栏对话编号。将该对话编号粘贴在下面的文本框中。"
            )
    st.chat_input(disabled=True)
