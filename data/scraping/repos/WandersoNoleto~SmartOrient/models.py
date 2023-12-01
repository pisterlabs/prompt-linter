from datetime import datetime

from django.db import models

from guidances.models import Guidance


class GuidanceArticle(models.Model):
    title         = models.CharField(max_length=150, verbose_name="Título")
    file          = models.FileField(upload_to='guidance_articles/', verbose_name="Arquivo")
    date_uploaded = models.DateField(auto_now_add=True, verbose_name="Date de Upload")
    guidance      = models.ForeignKey(Guidance, on_delete=models.CASCADE, verbose_name="Orientação")

    def title_format(self):
        self.title = self.title[:-4]

    def set_date_uploaded(self):
        date = datetime.now().strftime("%Y-%m-%d")
        self.date_uploaded = date

    def __str__(self):
        return self.title