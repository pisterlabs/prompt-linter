import os, magic

from langchain.docstore.document import Document
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter

from signal import SIGINT

class pdfLoader:
    pdfFilePaths    = []
    documentIndex   = []

    def __init__(self, persistdb=None, path=None, recurse=True, symlinks=True, signalHandler=None):
        self.path       = path
        self.recurse    = recurse       # if path is a directory, whether or not to recurse into subdirectories
        self.symlinks   = symlinks      # changes whether or not we resolve symlinks
        self.persistdb  = persistdb     # persistent memory store, for now Chroma/DuckDB
        self.ossignal   = signalHandler # optional signalHandler from OS, to interrupt long processes gracefully

        print("pdfLoader_init(): successfully initialized pdfLoader class")

    def queueFile(self, filePath=None):
        if filePath is None:
            filePath = self.path

        if self.symlinks:
            filePath = os.path.realpath(filePath)

        if filePath.endswith(".pdf"):
            # Check the application/mimetype of the provided file with the 'magic' library
            fileType = magic.from_file(filePath, mime=True)

            if fileType == "application/pdf":
                self.pdfFilePaths.append(filePath)
            else:
                print("pdfLoader_queueFile: " + filePath + ":" + fileType + " - skipping")
                return False

        return True

    def queueDirectory(self, dirPath=None):
        if dirPath is None:
            dirPath = self.path

        for file in os.listdir(dirPath):
            if (self.ossignal != None) and (self.ossignal.get_last_signal() == SIGINT):
                print("pdfLoader_queueDirectory: caught signal - exiting gracefully")
                self.emptyQueue()
                return True

            filePath = os.path.join((dirPath), file)

            if os.path.isfile(filePath):
                self.queueFile(filePath)

            if os.path.isdir(filePath):
                if self.recurse:
                    self.queueDirectory(filePath)
                else:
                    print("pdfLoader_queueDirectory: skipping directory " + filePath)

        return True

    def addPathToQueue(self, processPath=None):
        ret = False

        if processPath is None:
            print("pdfLoader_addToQueue(): invalid path or path not set - unable to continue")
            return False

        if os.path.isfile(processPath):
            ret = self.queueFile(processPath)

        if os.path.isdir(processPath):
            ret = self.queueDirectory(processPath)

        return ret

    def processQueue(self):
        if self.pdfFilePaths is []:
            print("pdfLoader_processQueue: empty queue - call process() first")
            return False

        for file in self.pdfFilePaths:
            if (self.ossignal != None) and (self.ossignal.get_last_signal() == SIGINT):
                print("pdfLoader_processQueue: caught signal - exiting gracefully")
                self.emptyQueue()
                return True

            print("pdfLoader_processQueue: processing " + file)
            try:
                loader = PyMuPDFLoader(file)
                data = loader.load()

                if data is not []:
                    dataChunks = []
                    dataSplitter = NLTKTextSplitter(chunk_size=768)
                    for page in data:
                        for chunk in dataSplitter.split_text(page.page_content):
                            dataChunks.append(Document(page_content=chunk, metadata=page.metadata))

                    self.persistdb.add_documents(dataChunks)
                else:
                    print("  -> bad document, or document is all images - cannot process")
                
            except Exception as error:
                print("  ~> " + file + ": " + str(error))
                continue

        return True

    def emptyQueue(self):
        self.dataIndex = []
        self.pdfFilePaths = []
        print("pdfLoader_emptyQueue: emptied queue")
        return

    def storeQueue(self):
        if self.documentIndex is []:
            print("pdfLoader_storeQueue: no documents loaded into index - run processQueue() first")
            return False
        self.persistdb.persist()

