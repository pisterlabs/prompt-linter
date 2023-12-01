from rest_framework import viewsets, status
from rest_framework.response import Response

from langchain.chains.question_answering import load_qa_chain

from apps.documents.vectorstore import get_vectorstore_for_chains
from apps.documents.models import DocumentCollection
from apps.chats.llm import get_embeddings_model, get_llm


class MessageViewSet(viewsets.ViewSet):
    def create(self, request):
        '''
        Builds the QA chain from a query and a list of documents, then runs the chain to get an answer.
        This is the bread and butter of the "chat your documents" feature.
        '''
        query = request.data.get('query')
        collection_id = request.data.get('collection_id')
        collection_name = request.data.get('collection_name')

        if query is None or (collection_id is None and collection_name is None):
            return Response(
                {"error": "query, and collection_id or collection_name parameters, are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        payload = None
        status_code = None

        try:
            # If a collection id is provided, use that to get the collection name.
            if collection_id:
                collection_name = DocumentCollection.objects.get(id=collection_id).name
                if not collection_name:
                    raise Exception('DocumentCollection not found.')

            docsearch = get_vectorstore_for_chains(get_embeddings_model(), collection_name)
            llm = get_llm()
            query = query.strip()
            chain = load_qa_chain(llm, chain_type='stuff')
            docs = docsearch.similarity_search(query)
            answer = chain.run(input_documents=docs, question=query).strip()
            payload = {
                'query': query,
                'answer': answer,
            }
            status_code = status.HTTP_200_OK
        except Exception as e:
            payload = {
                'error': str(e)
            }
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        finally:
            return Response(payload, status=status_code)
