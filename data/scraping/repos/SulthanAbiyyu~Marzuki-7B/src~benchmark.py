from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager

class Chatbot:
    def __init__(self, model_path, n_gpu_layers=256, n_batch=128, n_ctx=512, max_tokens=2048, verbose=True, repeat_penalty=1.18, temperature=0.5):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.repeat_penalty = repeat_penalty
        self.temperature = temperature
        self.template = """\
        Anda adalah chatbot di Universitas Brawijaya. \
        Tugas anda adalah menjawab pertanyaan dari tamu. \
        Gunakan bahasa yang sopan dan formal. \
        Pertanyaan: {question}
        Jawaban: 
        """
        self.template_2 = """\
        Anda adalah chatbot di Universitas Brawijaya. \
        Tugas anda adalah menjawab pertanyaan dari tamu. \
        Gunakan bahasa yang sopan dan formal. \
        Jika tamu menanyakan hal diluar tentang Universitas Brawijaya, \
        maka jawab dengan ‘Maaf, saya tidak dapat menjawab pertanyaan tersebut.’
        Pertanyaan: {question}
        Jawaban:    
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["question"])
        self.prompt_2 = PromptTemplate(template=self.template_2, input_variables=["question"])
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_ctx=self.n_ctx,
            verbose=self.verbose,
            repeat_penalty=self.repeat_penalty,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=False)
        self.llm_chain2 = LLMChain(prompt=self.prompt_2, llm=self.llm, verbose=False)

    def answer(self, question):
        return self.llm_chain.run(question)

    def answer_2(self, question):
        return self.llm_chain2.run(question)

if __name__ == "__main__":
    model_path = "../../models/merak-7B-v3/Merak-7B-v3-model-q8_0.gguf"
    pertanyaans = [
        "saya biyu, kamu siapa?",
        "bagaimana caranya daftar UB?",
        "Jelaskan kenapa pelangi ada 7 warna!"
    ]
    chatbot = Chatbot(model_path=model_path)
    a1 = chatbot.answer(pertanyaans[0])
    a2 = chatbot.answer(pertanyaans[1])
    a3 = chatbot.answer(pertanyaans[2])

    print("Template 1")
    print("Q1: ", pertanyaans[0])
    print("A1: ", a1)
    print("—- Pertanyaan Deskriptif")
    print("Q2: ", pertanyaans[1])
    print("A2: ", a2)
    print("—- Pertanyaan OOT")
    print("Q3: ", pertanyaans[2])
    print("A3: ", a3)

    a1 = chatbot.answer_2(pertanyaans[0])
    a2 = chatbot.answer_2(pertanyaans[1])
    a3 = chatbot.answer_2(pertanyaans[2])

    print("Template 2")
    print("Q1: ", pertanyaans[0])
    print("A1: ", a1)
    print("—- Pertanyaan Deskriptif")
    print("Q2: ", pertanyaans[1])
    print("A2: ", a2)
    print("—- Pertanyaan OOT")
    print("Q3: ", pertanyaans[2])
    print("A3: ", a3)
