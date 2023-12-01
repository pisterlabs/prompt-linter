"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the discussion is about the video content.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""""""You are an AI assistant designed for answering questions about a video.
You are given a document and a question, the document records what people see and hear from this video.
Try to connet these information and provide a conversational answer.
Question: {question}
=========
{context}
=========
""""""
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #video_inp {min-height: 100px}
      #chatbox {min-height: 100px;}
      #header {text-align: center;}
      #hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """