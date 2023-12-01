from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import logout
from django.http import HttpResponse
from django.urls import reverse
from django.utils import timezone
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.storage import default_storage

from chat.forms import MessageForm, UploadForm, QueryForm, QAForm, SettingsForm
from chat.models import Message, User, Chat, DocumentChunk, Document, UserSettings
from chat.llm_utils.vertex import (
    gcp_embeddings,
    get_docs_chunks_by_embedding,
    get_qa_response,
    text_llm,
    summarize_chain,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    text_splitter,
)

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.docstore.document import Document as LcDocument
from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI
from langchain.document_loaders import TextLoader, PyPDFLoader

from tempfile import NamedTemporaryFile
import os
import numpy as np


class IndexView(LoginRequiredMixin, TemplateView):
    template_name = "index.jinja"


def chat_response(request, chat_id):
    messages = (
        Message.objects.filter(chat_id=chat_id)
        .order_by("timestamp")
        .prefetch_related("chat")
    )
    user_settings = request.user.settings
    is_chat_model = "chat" in user_settings.model_name
    if is_chat_model:
        system_prompt = user_settings.system_prompt
        if system_prompt is None:
            system_prompt = settings.CHAT_SYSTEM_PROMPT
        chat_messages = [
            SystemMessage(
                content=system_prompt,
            )
        ]
        for i, message in enumerate(messages):
            if message.is_bot:
                chat_messages.append(AIMessage(content=message.message))
            elif i > 0 and not messages[i - 1].is_bot:
                chat_messages[-1].content += "\n" + message.message
            else:
                chat_messages.append(HumanMessage(content=message.message))
    else:
        # Only one prompt for non-chat models
        chat_messages = messages.first().message
    if not is_chat_model and len(messages) > 2:
        bot_message = Message.objects.create(
            message="Non-chat models only support one prompt. Please start a new chat or switch to a chat model.",
            chat_id=chat_id,
            is_bot=True,
        )
    else:
        is_code_model = "code" in user_settings.model_name
        llm_class = ChatVertexAI if is_chat_model else VertexAI
        max_tokens = 2048 if is_code_model else 1024
        if "32k" in user_settings.model_name:
            max_tokens = 8192  # Not sure why this is imposed, but it is
        max_tokens = min(max_tokens, user_settings.max_output_tokens)
        llm = llm_class(
            model_name=user_settings.model_name,
            max_output_tokens=max_tokens,
            temperature=user_settings.temperature,
        )
        num_tokens = None
        if user_settings.debug:
            if is_chat_model:
                num_tokens = llm.get_num_tokens_from_messages(chat_messages)
            else:
                num_tokens = llm.get_num_tokens(chat_messages)
        llm_response = llm(chat_messages)
        bot_message = Message.objects.create(
            message=llm_response.content if is_chat_model else llm_response,
            chat_id=chat_id,
            is_bot=True,
        )
    chat = Chat.objects.get(id=chat_id)
    add_chat_title = False
    # Title the chat and add it to the sidebar if there's sufficient context to do so
    if chat.title == "" and (
        len(messages) > 2
        or not is_chat_model
        or len(" ".join([m.content for m in chat_messages])) > 300
    ):
        add_chat_title = True
        chat.title = summarize_chat(chat_id)
        chat.save()

    context = {
        "message": bot_message,
        "add_chat_title": add_chat_title,
        "chat": chat,
    }

    if user_settings.debug:
        context.update(
            {
                "num_tokens": num_tokens,
                "debug": True,
            }
        )
    return render(
        request,
        "fragments/waiting_message.jinja",
        context,
    )


class ChatView(LoginRequiredMixin, TemplateView):
    # GET method returns the entire chat page
    # POST method creates message
    # and returns HTML fragment with message + "LLM typing" indicator
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_tab"] = "chat"
        context["form"] = MessageForm()
        try:
            user_settings = self.request.user.settings
        except ObjectDoesNotExist:
            user_settings = UserSettings.objects.create(user=self.request.user)
        if user_settings.system_prompt is None:
            user_settings.system_prompt = settings.CHAT_SYSTEM_PROMPT
        context["settings_form"] = SettingsForm(instance=user_settings)
        context["default_system_prompt"] = settings.CHAT_SYSTEM_PROMPT
        # Get non-empty chats for user
        user_chats = (
            Chat.objects.filter(user=self.request.user)
            .order_by("-timestamp")
            .prefetch_related("message_set")
        )
        user_chats = [chat for chat in user_chats if chat.message_set.count() > 1]
        context["chat"] = Chat.objects.get(id=kwargs["chat_id"])
        # Summarize chats into chat.title if not already set
        for chat in user_chats:
            if chat.title == "":
                chat.title = summarize_chat(chat.id)
                chat.save()
        context["user_chats"] = user_chats
        return context

    # Validate that request.user is Chat.user
    def dispatch(self, request, *args, **kwargs):
        if "chat_id" in kwargs:
            chat = Chat.objects.get(id=kwargs["chat_id"])
            if not chat.user == request.user:
                return HttpResponse(status=403)
        return super().dispatch(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        form = MessageForm(request.POST)
        if form.is_valid():
            user_message = Message.objects.create(
                message=form.cleaned_data["message"],
                chat_id=kwargs["chat_id"],
            )
            return render(
                request,
                "fragments/waiting_message.jinja",
                {
                    "message": user_message,
                    "chat_id": kwargs["chat_id"],
                    "waiting": True,
                },
            )
        else:
            return self.render_to_response({"form": form})

    template_name = "chat.jinja"


class NewChatView(ChatView):
    """
    Creates a new chat on GET and renders the chat template
    """

    def get(self, request, *args, **kwargs):
        # Create a new chat
        chat = Chat.objects.create(user=request.user)
        # Redirect to the chat view
        return redirect("chat", chat_id=chat.id)


def logout_view(request):
    logout(request)
    return redirect("index")


def warmup(request):
    return HttpResponse(status=200)


def chat_settings(request):
    # Update user settings
    if request.method == "POST":
        form = SettingsForm(request.POST)
        if form.is_valid():
            user_settings = request.user.settings
            user_settings.system_prompt = form.cleaned_data["system_prompt"]
            user_settings.model_name = form.cleaned_data["model_name"]
            user_settings.max_output_tokens = form.cleaned_data["max_output_tokens"]
            user_settings.temperature = form.cleaned_data["temperature"]
            user_settings.debug = form.cleaned_data["debug"]
            user_settings.save()
            response = HttpResponse(status=200)
            # Add message to response to be displayed by HTMX
            response["HX-Trigger"] = "settings-updated"
            return response


def delete_chat(request, chat_id, current_chat=None):
    # HTMX delete route
    # Delete chat
    chat = Chat.objects.get(id=chat_id)
    if not chat.user == request.user:
        return HttpResponse(status=403)

    chat.delete()
    # Is this the currently open chat? If so, redirect away
    if current_chat == "True":
        response = HttpResponse()
        response["HX-Redirect"] = reverse("index")
        return response
    return HttpResponse(status=200)


def summarize_chat(chat_id):
    chat_text = "\n".join(
        [message.message for message in Message.objects.filter(chat_id=chat_id)[:5]]
    )
    if chat_text == "":
        return "Empty chat"
    if len(chat_text) > 500:
        chat_text = chat_text[:500]
    prompt = f"""
    Write a concise title (1-4 words) to the following chat. You must respond with at least one word:
    {chat_text}
    TITLE: """
    return text_llm(prompt)


class DocumentsView(LoginRequiredMixin, TemplateView):
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["active_tab"] = "documents"
        context["upload_form"] = UploadForm()
        context["documents"] = Document.objects.filter(user=self.request.user).order_by(
            "-uploaded_at"
        )
        context["query_form"] = QueryForm()
        context["qa_form"] = QAForm()
        return context

    def delete(self, request, *args, **kwargs):
        document = Document.objects.get(id=kwargs["doc_id"])
        if document.user == request.user:
            default_storage.delete(document.file.name)
            document.delete()
            return HttpResponse(status=200)
        else:
            return HttpResponse(status=403)

    def post(self, request, *args, **kwargs):
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["file"]
            file_bytes = uploaded_file.file
            temp_file = NamedTemporaryFile(delete=False)
            temp_file.write(file_bytes.read())
            temp_file.seek(0)
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_file.name)
            else:
                loader = TextLoader(temp_file.name, encoding="utf8")
            docs = loader.load()
            temp_file.close()
            os.unlink(temp_file.name)
            text = "\n\n".join([doc.page_content for doc in docs])
            # Delete any existing documents with the same original_filename
            # Document.objects.filter(
            #     user=request.user, original_filename=uploaded_file.name
            # ).delete()
            instance = Document(
                file=uploaded_file,
                user=request.user,
                uploaded_at=timezone.now(),
                chunk_overlap=CHUNK_OVERLAP,
                chunk_size=CHUNK_SIZE,
                title=uploaded_file.name,
                original_filename=uploaded_file.name,
                text=text,
            )
            instance.save()
            return render(
                request, "fragments/document_row.jinja", {"doc": instance, "new": True}
            )
        else:
            return self.render_to_response({"form": form})

    template_name = "documents.jinja"


def summary(request, doc_id):
    doc = Document.objects.get(id=doc_id)
    title = generate_title(doc)
    summary = summarize(doc)
    # Add the document filename to the summary for embedding
    summary_for_embedding = title + "\n" + doc.file.name + "\n\n" + summary
    summary_embedding = gcp_embeddings.embed_documents([summary_for_embedding])[0]
    Document.objects.filter(id=doc_id).update(
        summary=summary,
        title=title,
        summary_embedding=summary_embedding,
    )
    doc = Document.objects.get(id=doc_id)
    return render(request, "fragments/summary.jinja", {"doc": doc})


def summarize(document):
    docs = [
        LcDocument(
            page_content=document.file.name + document.text[:10000],
        )
    ]
    summary = summarize_chain.run(docs)
    # Sometimes the summarizer returns an empty string
    if summary == "":
        summary = document.file.name
    return summary


def generate_title(document):
    if len(document.text) < 10:
        return document.file.name
    text = f"Filename: {document.file.name}\n\nContent (first 5,000 characters): {document.text[:5000]}"
    prompt = f"""
    Write a concise title (1-5 words) for the following document. You must respond with at least one word:
    {text}
    1-5 WORD TITLE: """
    return text_llm(prompt)[:255]


def full_text(request, doc_id):
    doc = Document.objects.get(id=doc_id)
    return render(request, "fragments/full_text.jinja", {"doc": doc})


def generate_embeddings(request, doc_id):
    doc = Document.objects.filter(id=doc_id).first()
    chunks = text_splitter.split_text(doc.text)
    for i, chunk in enumerate(chunks):
        DocumentChunk.objects.create(
            document=doc,
            chunk_number=i,
            text=chunk,
        )
    chunks = doc.chunks.all().order_by("chunk_number")
    texts = chunks.values_list("text", flat=True)
    # Add some extra context to each chunk, e.g. filename, title, summary, tags
    texts = [
        """Filename: {doc.file.name}
Document title: {doc.title}
Document summary: {doc.summary}\n
Chunk content: {text}"""
        for text in texts
    ]
    embeddings = gcp_embeddings.embed_documents(texts)
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i]
    DocumentChunk.objects.bulk_update(chunks, ["embedding"])
    Document.objects.filter(id=doc_id).update(
        mean_embedding=np.mean(embeddings, axis=0)
    )
    doc = Document.objects.filter(id=doc_id).prefetch_related("chunks").first()
    return render(request, "fragments/embeddings_preview.jinja", {"doc": doc})


def query_embeddings(request):
    query = request.GET.get("query")
    if query is None:
        return HttpResponse(status=400)
    documents_by_summary, chunks_by_embedding = get_docs_chunks_by_embedding(
        request, query
    )
    return render(
        request,
        "fragments/query_results.jinja",
        {
            "chunks": chunks_by_embedding,
            # "documents_by_mean": documents_by_mean,
            "documents_by_summary": documents_by_summary,
        },
    )


def qa_embeddings(request):
    query = request.GET.get("query")
    if query is None:
        return HttpResponse(status=400)
    documents_by_summary, chunks_by_embedding = get_docs_chunks_by_embedding(
        request, query, max_distance=0.5
    )
    response = get_qa_response(
        query,
        [
            LcDocument(page_content=doc.summary, metadata={"source": doc.file.name})
            for doc in documents_by_summary
        ]
        + [
            LcDocument(
                page_content=chunk.text, metadata={"source": chunk.document.file.name}
            )
            for chunk in chunks_by_embedding
        ],
    )
    if "\nSOURCES" in response:
        response = (
            response[: response.index("\nSOURCES")].replace("\n", "<br>")
            + "<br><br><b>Sources</b>: "
            + response[response.index("\nSOURCES") + 9 :].replace("\n", "<br>")
        )
    return HttpResponse(response)
