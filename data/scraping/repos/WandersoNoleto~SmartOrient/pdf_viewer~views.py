from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render

from library.models import GuidanceArticle


def view_pdf(request, id):
    article = get_object_or_404(GuidanceArticle, id=id)

    context = {
        'article': article,
    }

    return render(request, "pdfViewer.html", context)