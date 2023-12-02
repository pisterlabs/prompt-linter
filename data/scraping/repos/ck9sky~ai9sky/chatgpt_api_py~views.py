from django.conf import settings
from django.contrib import messages
from django.urls import reverse_lazy
from django.views import generic

from openai import OpenAI   # The openai Python library. 11/11/23

## from . import models
from . import forms


class ChatGPT_API_PY_Test1(generic.FormView):
    """ This app (chatgpt_api_py) is more like a "traditional" Django app because Python openai Library is used (instead of relying
        mostly on JavaScript to use CHATGPT API). HOWEVER, if you ever want this app (as of Nov 2023), to have identical features as
        chatgpt_api_js' (the "mostly JS" app for ChatGPT API access), you will need to replicate the advantage 'chatgpt_api_js' has,
        where 'chatgpt_api_js' uses no page reloads because of GET/POST's to remote "Django" database: THIS APP CURRENTLY CANNOT
        SHOW A TRUE CHATLOG: PRIOR CHATGPT PROMPT/RESPONSE SESSION IS "THROWN AWAY" WITH NEXT PAGE RELOAD. 11/17/23
        -------------------------------------------------------------------------------------------------------------------------
       OPENAI DOCS FOR CHATGPT API (Not as crucial to read docs given we use Python openai Library?):
       -- openai.com (may need to login, browse back to openai.com)
       -- API | Docs (menu)
       -- API reference (tab)
       -- ENDPOINTS | Chat (side bar)
       -- Create chat completion ... etc.
    """
    form_class = forms.ChatGPT_API_PY_Test1_Prompt_Form
    template_name = "chatgpt_api_py/chatgpt_api_py.html"
    plus_context = dict()

    def get_success_url(self, *args, **kwargs):
        """ Does not leave the page """
        return reverse_lazy("chatgpt_api_py:chatgpt_api_py_test1")

    def get_context_data(self, **kwargs):
        """ Setting plus_context vars back to null string does not seem needed in this app(?) (chatgpt_api_py),
            but was needed in dalle_api_py(?) to avoid odd "caching error" (read discussion in view DALLE_API_PY_Test1).
            11/18/23 """
        context = super().get_context_data(**kwargs)
        if self.plus_context:
            if 'prompt' in self.plus_context:
                context['prompt'] = self.plus_context['prompt']
                self.plus_context['prompt'] = ""      # Prevent caching error? (Probably not in this app...)
            else:
                context['prompt'] = ""

            if 'message' in self.plus_context:
                context['message'] = self.plus_context['message']
                self.plus_context['message'] = ""     # Prevent caching error? (Probably not in this app...)
            else:
                context['message'] = "Thinking"
        else:
            context['prompt'] = ""
            context['message'] = "Thinking"
        return context

    def form_valid(self, form):
        """ We will add code to show this message (or remove this message if not necessary. 11/11/23
        """
        question = form.cleaned_data['prompt']
        if settings.NULL_STR.__eq__(question):
            messages.error(
                self.request,
                "You Must Enter a Question!")  # BUT as of Nov 2023, site does not display Django messages (11/17/23
            form.add_error('prompt', True)
            return self.form_invalid(form)
        """
            Now we use Python library module openai. 11/11/23
            -----------------------------------------------------------------
            https://platform.openai.com/docs/libraries 
            https://platform.openai.com/docs/api-reference?lang=python  
            https://github.com/openai/openai-python
        """
        client = OpenAI()
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )

        self.plus_context['prompt'] = question
        self.plus_context['message'] = chat_completion.choices[0].message.content  # Send AI reply 'message' to template!
        return super().form_valid(form)

