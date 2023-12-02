from django.urls import path
from langchainapp.views import *

urlpatterns = [
    path('url_embedding', EmbeddingURL.as_view(), name="url_embedding"),
    path('pdf_embedding', EmbeddingPDF.as_view(), name="pdf_embedding"),
    path('csv_embedding', EmbeddingCSV.as_view(), name="csv_embedding"),
    path('txt_embedding', EmbeddingTXT.as_view(), name="txt_embedding"),
    path('remove_embedding', RemoveEmbedding.as_view(), name="remove_embedding"),
    path('save_attr', LangAttr.as_view(), name="save_attr"),
    path('slack_bot', LangSlack.as_view(), name="slack-bot"),
    path('get_attr', GetLangAttr.as_view(), name="get_attr"),
    path('mail_detect', MailDetect.as_view(), name="mail_detect"),
    path('chat', CHAT.as_view(), name="chat"),
    path('get_init_file_list', GetEmbeddingData.as_view(), name="get_embedding_data"),
]
