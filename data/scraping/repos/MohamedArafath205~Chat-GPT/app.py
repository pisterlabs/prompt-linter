import streamlit as st
import openai


st.set_page_config(page_title="Chat GPT", page_icon=":crown:", layout="wide")

# ---- Header ----
def main():
  st.session_state.setdefault("logs", [])

if __name__ == "__main__": 
  main()

  st.header("Chat GPT")

  st.subheader("""
              HI :wave:, 
              I am Mohamed Arafath an AIML student at SRM Institue of Science and Technology. 
              I am a Jack of all cards! 
              I know Machine Learning and Intrested to learn more about web developement""")
  st.title("This is my [GitHub](https://github.com/MohamedArafath205) follow me!")

  title = st.text_input("Ask me anything...", key="input")

  if "logs" not in st.session_state:
    st.session_state.logs = []



  openai.api_key = "YOUR_API_KEY"

  if(title != ""):
    with st.spinner("Generating response..."):
      response = openai.Completion.create(
      model="text-davinci-003",
      prompt=title,
      temperature=0,
      max_tokens=60,
      top_p=1,
      frequency_penalty=0.5,
      presence_penalty=0
    )
      
      
  if st.button('Enter'):
      bot_response = response.choices[0].text.strip()
      st.session_state.logs.append(bot_response)
      message_box = f"<div style='background-color:#f2f2f2; padding:10px; border-radius:10px; margin-bottom:10px; height:200px; overflow-y:scroll; color:black;'>{bot_response}</div>"
      st.markdown(message_box, unsafe_allow_html=True)
      input_text = ""
      

  if st.button("Clear conversation"):
    st.session_state.logs = []
    
  bot_logs = [log for log in st.session_state.logs if log != ""]
  if bot_logs:
    with st.expander("Conversation History"):
      conversation_history = "\n\n".join(bot_logs[::-1])
      st.text_area(" ", value=conversation_history, height=500)
