from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from guidances.models import Guidance
from library.models import GuidanceArticle


def save_guidance_article(request, guidance_id):
    if request.method == "POST":
        title    = request.POST.get("title")
        file     = request.FILES.get("file")
        guidance = get_object_or_404(Guidance, id=guidance_id)

        article = GuidanceArticle(
            title    = title,
            file     = file,
            guidance = guidance
        )

        article.title_format
        article.set_date_uploaded
        article.save()

    return redirect(reverse('open_guidance', kwargs={'id': guidance_id}))

