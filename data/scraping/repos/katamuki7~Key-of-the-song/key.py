import openai
import os
import streamlit as st
from dotenv import load_dotenv

openai.api_key = os.environ.get("OPENAI_API_KEY")

oto_ary=["C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
         "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
         "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
         "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
         "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6", "C7"]

model = "gpt-3.5-turbo"

my_saitei_cnt = 0
saitei_cnt = 0
my_oniki_len = 0
oniki_len = 0
my_oniki = []
oniki = []

def oto_search(oto):
    search_cnt = 0
    for i in oto_ary:
        if oto != i:
            search_cnt += 1
        else:
            break
    return search_cnt

st.title("キー調整ページ")

if 'ans' not in st.session_state:
    st.session_state.ans = None


question = st.text_input("曲名を入れてください")
if st.button("検索"):
    if st.session_state["ans"] == None:
        res = openai.ChatCompletion.create(
            model=model,
            messages = [
                {"role": "system", "content": "あなたは曲名を入れるとその曲の最低音と最高音を教えるロボットです。アーティストの名前も出してください。"},
                {"role": "user", "content": "Q: " + question},
            ])
        st.session_state["ans"] = res.choices[0]["message"]["content"]
        st.write(st.session_state["ans"])
    else:
        st.write(st.session_state["ans"])

if st.button("検索リセット"):
    st.session_state["ans"] = None

st.markdown("### 自分が出せるキー")
col1, col2 = st.columns(2)
my_saitei = col1.selectbox("自分の最低音", (oto_ary))
my_saikou = col2.selectbox("自分の最高音", (oto_ary))

st.markdown("### 合わせる曲のキー")
col3, col4 = st.columns(2)
saitei = col3.selectbox("曲の最低音", (oto_ary))
saikou = col4.selectbox("曲の最高音", (oto_ary))

st.text("最低音と最高音の選択か逆になっていないか。正しく選択できているか確認してください")

if st.button("調べる"):
    my_oniki = []
    oniki = []
    my_saitei_cnt = oto_search(my_saitei)
    saitei_cnt = oto_search(saitei)

    for i in range(my_saitei_cnt, 61):
        my_oniki.append(oto_ary[i])
        if my_saikou == oto_ary[i]:
            break

    for i in range(saitei_cnt, 61):
        oniki.append(oto_ary[i])
        if saikou == oto_ary[i]:
            break

    my_oniki_len = len(my_oniki)
    oniki_len = len(oniki)
    st.markdown("### こちらのキーに変更してみてください")

    #音域が足りない場合
    if my_oniki_len < oniki_len:
        y = my_saitei_cnt - saitei_cnt
        if abs(y) < 6:
            y = str(y)
        elif abs(y%12) == 6:
            y = "+6か-6"
        elif y > 6:
            y = y - 12
            y = str(y)
        elif abs(y) % 12 * -1 <= -7:
            y = abs(y) % 12 * -1 + 12
            y = str(y)
        else:
            y = abs(y) % 12 * -1
            y = str(y)
        st.warning('あなたの音域が足りません。曲とあなたの最低音が一致するキーを表示しています。')
        st.subheader(y)
    else: #足りるとき
        syusei = 0
        y = my_oniki_len - oniki_len
        if y == 0:
            y = str(y)
        else:
            syusei = y // 2
            y = my_saitei_cnt - saitei_cnt + syusei
            if abs(y) < 6:
                y = str(y)
            elif abs(y%12) == 6:
                y = "+6か-6"
            elif y > 6:
                y = y - 12
                y = str(y)
            elif abs(y) % 12 * -1 <= -7:
                y = abs(y) % 12 * -1 + 12
                y = str(y)
            else:
                y = abs(y) % 12 * -1
                y = str(y)
        st.info('あなたの音域は足りています。曲の音域があなたの音域の中心になるキーを表示しています。')
        st.subheader(y)
#print(my_saitei_cnt, saitei_cnt, my_oniki_len, oniki_len,"my_oniki, oniki")
