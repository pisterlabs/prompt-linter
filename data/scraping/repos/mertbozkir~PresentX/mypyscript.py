import os 
import openai
import streamlit as st
import time

api_key = 'api_key'
openai.api_key = api_key

st.title('GPT-3 Hackathon')
st.text('Instructions : \n User: "description of output"\n Code:')
text_input =  st.text_area(label='description',height=100)

prompt_key = r"\nUser: Make a detailed and informative presentation on Glioblastoma. First page should be title page, with title Malignant Brain Tumor.\nCode: \documentclass{beamer}\mode<presentation> {\usetheme{Madrid}\usecolortheme{wolverine}}\title[Short title]{Malignant Brain Tumor}\begin{document}\begin{frame}\titlepage\end{frame}\begin{frame}{Glioblastoma} \begin{block}{Introduction} \begin{itemize} \item Glioblastoma (GBM) is a type of cancerous brain tumor that originates from the glial cells, which are the cells that surround and support neurons. \item The prognosis for GBM is poor with a mean survival time of 12–15 months from diagnosis. \end{itemize} \end{block} \begin{block}{Causes} \begin{itemize} \item The exact cause of GBM is unknown, but it is thought to be caused by a combination of genetic and environmental factors.\item Glioblastoma is known to occur with exposure to ionizing radiation, benzene, aniline, certain chemicals, and asbestos.\end{itemize} \end{block} \begin{block}{Symptoms} \begin{itemize} \item The symptoms of GBM are related to both the location and the size of the tumor. \item The symptoms are described as a spectrum ranging from completely asymptomatic to very serious.\end{itemize} \end{block} \end{frame}\end{document}\n\nUser: I need a slide consisting of table on Epoch and Accuracy\nCode:\documentclass{beamer}\mode<presentation> { \usetheme{Madrid} }\begin{document} \begin{frame} \frametitle{Epochs} \begin{table} \begin{tabular}{l l l} \toprule \textbf{Epoch} & \textbf{Accuracy} \\ \midrule Epoch 10 & 53.2\% \\ Epoch 20 & 60\% \\ Epoch 30 & 68.8\% \\ \bottomrule \end{tabular} \caption{Epochs and Accuracy} \end{table} \end{frame} \end{document}\n\nUser: Make a presentation with one slide that shows Einstein's equation relating mass to energy.\nCode: \documentclass{beamer}\mode<presentation> {\usetheme{Madrid}\usecolortheme{wolverine}}}\begin{document}\begin{frame}\frametitle{Einstein's equation for Special Relativity}\begin{theorem}[Mass--energy equivalence]$E = mc^2$\end{theorem}\end{frame}\end{document}\n\nUser: Give me a slide on Euler Equations\nCode: \documentclass{beamer} \mode<presentation> { \usetheme{Madrid}\n} \begin{document} \begin{frame} \frametitle{Euler Equations} {Euler Equations} \begin{equation}e^{i\pi}+1=0\end{equation} \begin{equation}i\,\!=\,\sqrt{-1} \end{equation} \begin{equation}e^{\pi i}+1=0\end{equation} \begin{equation}i^{(1)}=\sqrt{-1} \end{equation} \begin{equation}e^{i\pi}=-1\end{equation} \begin{equation}i^{(2)}=-1\end{equation} \begin{equation}e^{-i\pi}=-1\end{equation} \begin{equation}i^{(3)}=-1\end{equation} \end{frame} \end{document}\n\nUser: Create a presentation with the title Machine Learning & AI. First page should be title page. Second page should be an overview slide, third page should have examples of application of modern Machine Learning. \nCode: \documentclass{beamer}\mode<presentation> {\usetheme{Madrid}\usecolortheme{wolverine}}\title[Short title]{Machine Learning & AI}\begin{document}\begin{frame}\titlepage \end{frame}\begin{frame}\frametitle{Overview}\tableofcontents\end{frame}\section{Sub disciplines within Machine Learning} \subsection{Computer Vision, Natural Language Processing, Reinforcement Learning}\begin{frame}\frametitle{Modern Machine Learning Applications}\begin{block}{Reinforcement Learning}{Autonomous vehicles and self-driving cars}\end{block}\begin{block}{Computer Vision}{Automatic face recognition}\end{block}\begin{block}{Natural Language Processing}Using language models to auto generate programming code from natural instructions\end{block}\end{frame}\end{document}\n\n:"
if len(text_input)> 0:
	response = openai.Completion.create(
		  engine="davinci",
		  prompt=prompt_key+text_input,
		  temperature=0.7,
		  max_tokens=736,
		  top_p=1,
		  frequency_penalty=0,
		  presence_penalty=0,
		  stop=["\n", "User:", "Code:"])
	time.sleep(10)
	st.text('Super convenient code: ')
	# st.write(response)
	st.write(response['choices'][0]['text'])
else:
	print('NoInputYet')


