from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base_operator import BaseOperator

from ai_context import AiContext


class SplitData(BaseOperator):
    @staticmethod
    def declare_name():
        return 'Recursively Split Text'
    
    @staticmethod
    def declare_category():
        return BaseOperator.OperatorCategory.MANIPULATE_DATA.value

    @staticmethod
    def declare_parameters():
        return [
            {
                "name": "chunk_size",
                "data_type": "integer",
                "placeholder": "Enter Chunk Size (Optional: Default is 2000)"
            },
            {
                "name": "chunk_overlap",
                "data_type": "integer",
                "placeholder": "Enter Chunk Overlap (Optional: Default is 100)"
            }
        ]

    @staticmethod
    def declare_inputs():
        return [     
            {
                "name": "text",
                "data_type": "string",
            }
        ]

    @staticmethod
    def declare_outputs():
        return [
            {
                "name": "rts_processed_content",
                "data_type": "string",
            }
        ]

    def run_step(
            self,
            step,
            ai_context: AiContext
    ):
        params = step['parameters']
        split_text = self.process(params, ai_context)
        ai_context.set_output('rts_processed_content', split_text, self)
        ai_context.add_to_log("Successfully split text!")

    def process(self, params, ai_context):
        text = ai_context.get_input('text', self)
        formatted = self.split(params, ai_context, text)
        return formatted

    def split(self, params, ai_context, content):
        chunk_size = params.get('chunk_size', '2000')
        chunk_overlap = params.get('chunk_overlap', '100')

        if chunk_size:
            chunk_size = int(chunk_size)
        else:
            chunk_size = 2000

        if chunk_overlap:
            chunk_overlap = int(chunk_overlap)
        else:
            chunk_overlap = 100
        ai_context.add_to_log(f"Splitting text with {chunk_size} size and {chunk_overlap} overlap")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(content)
        return texts
