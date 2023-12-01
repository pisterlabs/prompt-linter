from PySide6.QtCore import QRunnable, Signal, QObject
from PySide6.QtWidgets import QListWidgetItem
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from spgpt.pdf import retrieve_pdf_data

class ImportPDFSignals(QObject):
    # finished = Signal()
    error = Signal(str)
    data_acquired = Signal(FAISS)

class ImportPDF(QRunnable):

    # finished = Signal()
    error = Signal(str)
    data_acquired = Signal(FAISS)

    def __init__(self, pdf_link:str, embeddings:OpenAIEmbeddings, cache_dir:str):
        super().__init__()
        self._pdf_link = pdf_link
        self._embeddings = embeddings
        self._cache_dir = cache_dir

        self._signals = ImportPDFSignals()
        self.error = self._signals.error
        self.data_acquired = self._signals.data_acquired

    def run(self):
        try:
            faiss_db = retrieve_pdf_data(self._pdf_link, self._embeddings, self._cache_dir)
        except Exception as e:
            pass
            # self._item.setText(f'Error: {self._pdf_link}')
            # self.finished.emit()
            self.error.emit(repr(e))
        else:
            pass
            # self.finished.emit()
            self.data_acquired.emit(faiss_db)
            # self._item.setText(f'{self._pdf_link}')