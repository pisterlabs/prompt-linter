import tempfile
import os
from pathlib import Path
import pypdfium2 as pdfium
from toolz import partition_all
import pandas as pd
import numpy as np
from postgres_copy import CopyManager
import uuid
from tqdm import tqdm
from django.contrib.auth import get_user_model
from django.conf import settings
from django.utils import timezone
from django.db import models
import threading
import time
import cohere 
import docspace
from .search_index import search_index
from .utils import *


class Cluster(models.Model):
    cluster_id = models.IntegerField()
    description = models.TextField(blank=True, null=True,)

    def get_description(self):
        if self.description is None:
            chunks = Chunk.objects.filter(cluster=self).order_by('cluster_distance')
            if len(chunks) > 0:
                prompt = '\n\n'.join([x.summary for x in chunks[:4]]) + '\n\n'
                prompt += 'Provide a concise summary that consolidates the descriptions above.'

                co = cohere.Client(docspace.config['COHERE_API_KEY']) 
                r = co.generate( 
                    model='command-xlarge-20221108', 
                    prompt=prompt,
                    max_tokens=300, 
                    temperature=0,
                )
                self.description = r.generations[0].text
                self.save()
        return self.description

    def __str__(self):
        return f'Claim Type {self.cluster_id}'


class Chunk(models.Model):
    doc = models.ForeignKey('Document', on_delete=models.CASCADE)
    page = models.IntegerField()
    chunk_index = models.IntegerField()
    text = models.TextField(blank=True, null=True)
    clean_text = models.TextField(blank=True, null=True)

    summary = models.TextField(blank=True, null=True)
    summary_array = models.JSONField(blank=True, null=True)
    cluster = models.ForeignKey('Cluster', blank=True, null=True, on_delete=models.CASCADE)
    cluster_distance = models.FloatField(blank=True, null=True)
    similar_docs = models.JSONField(blank=True, null=True)

    objects = CopyManager()

    def get_summary(self):
        if self.summary is None:
            prompt = self.text.replace('__', ' ') + '\n\n'
            prompt += 'What claims is the plaintiff bringing against the defendant?'

            co = cohere.Client(docspace.config['COHERE_API_KEY']) 
            r = co.generate( 
                model='command-xlarge-20221108', 
                prompt=prompt,
                max_tokens=300, 
                temperature=0,
            )
            q1 = r.generations[0].text

            prompt = prompt + '\n\n' + q1 + '\n\nDescribe these laws.'
            r = co.generate( 
                model='command-xlarge-20221108', 
                prompt=prompt,
                max_tokens=300, 
                temperature=0,
            )
            q2 = r.generations[0].text

            self.summary = q1 + ' ' + q2
            self.save()
        return self.summary
    
    def get_summary_array(self):
        if self.summary_array is None:
            co = cohere.Client(docspace.config['COHERE_API_KEY']) 
            r = co.embed(texts=[self.summary])
            self.summary_array = r.embeddings[0]
            self.save()
        return self.summary_array

    def search(self, k=10):
        query = np.array([self.summary_array]).astype(np.float32)
        search_k = k + len(self.doc.chunks())
        results = search_index.search(query, search_k)[0]
        results = [(x['distance'], Chunk.objects.get(id=int(x['text']))) for x in results]
        results = [x for x in results if x[1].doc != self.doc][:k]
        return results
    
    def get_cluster(self):
        if self.cluster is None:
            results = self.search(1)
            if len(results) > 0:
                self.cluster = results[0][1].cluster
                self.cluster_distance = results[0][0]
                self.save()
        return self.cluster
    
    def get_similar_docs(self):
        if self.similar_docs is None:
            cluster = self.get_cluster()
            similar_docs = Chunk.objects.filter(cluster=cluster).exclude(doc=self.doc).order_by('cluster_distance').values_list('doc_id', flat=True).distinct()
            self.similar_docs = [str(x) for x in similar_docs[:8]]
            self.save()
        similar_docs = [uuid.UUID(x) for x in self.similar_docs]
        similar_docs = Document.objects.filter(id__in=similar_docs)
        return similar_docs


def pdf_path(instance, filename):
    return f'pdf/{instance.id}.pdf'

class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    pdf = models.FileField(upload_to=pdf_path, blank=True, null=True)
    text = models.TextField(blank=True, null=True)
    clean_text = models.TextField(blank=True, null=True)
    info = models.JSONField(default=dict)
    upload_date = models.DateTimeField(auto_now_add=True)
    upload_by = models.ForeignKey('auth.User', on_delete=models.CASCADE, blank=True, null=True)
    upload_session = models.CharField(max_length=255, blank=True, null=True)
    last_processed = models.DateTimeField(blank=True, null=True)
    public = models.BooleanField(default=False)

    def __str__(self):
        return self.name

    def load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            input_path = temp_dir / 'doc.file'
            
            with open(input_path, 'wb') as w:
                w.write(self.pdf.file.read())


            pdf = pdfium.PdfDocument(str(input_path))
            return pdf

    def process(self, max_chunks=20):
        Chunk.objects.filter(doc=self).delete()
        pdf = self.load()

        chunk_size = 256
        chunks = [{'tokens': [], 'page': 0, 'chunk_index': 0}]

        for page_index, page in enumerate(pdf):
            textpage = page.get_textpage()
            for rect in textpage.get_rectboxes():
                text = textpage.get_text_bounded(*rect)
                tokens = text.split()
                if len(tokens) + len(chunks[-1]['tokens']) < chunk_size:
                    chunks[-1]['tokens'] += tokens
                else:
                    chunks.append({
                        'tokens': tokens,
                        'page': page_index,
                        'chunk_index': len(chunks),
                    })

        all_tokens = []
        keep_next = 0
        terms = ['CLAIM', 'COUNT', 'CAUSE', 'Claim ', 'Count ', 'Cause ']
        consolidated_chunks = []
        for i, chunk in tqdm(list(enumerate(chunks))):
            all_tokens += chunk['tokens']
            text = ' '.join(chunk['tokens'])
            if any(term in text for term in terms):
                keep_next = max([keep_next, 3])
            if keep_next > 0:
                consolidated_chunks.append(chunk)
                keep_next -= 1
            
        consolidated_chunks = consolidated_chunks[:max_chunks]
        chunks = pd.DataFrame(consolidated_chunks)
        if len(chunks) > 0:
            chunks['doc_id'] = str(self.id)
            chunks['text'] = chunks['tokens'].apply(lambda x: ' '.join(x))
            chunks['clean_text'] = chunks['text'].apply(docspace.utils.clean_text)
            chunks = chunks.drop(columns=['tokens'])
            Chunk.objects.from_csv(df_to_file(chunks))
            
        self.text = ' '.join(all_tokens)
        self.clean_text = docspace.utils.clean_text(self.text)
        self.last_processed = timezone.now()
        self.save()

    def update_chunks(self, sleep=3, skip_similarity_matching=False):
        chunks = Chunk.objects.filter(summary_array__isnull=True, doc=self)

        def update_chunk(chunk):
            try:
                chunk.get_summary()
                chunk.get_summary_array()
                if not skip_similarity_matching:
                    chunk.get_cluster()
                    chunk.get_similar_docs()
            except Exception as e:
                print('error', e)
                time.sleep(5)

        if len(chunks) > 0:
            for chunk in chunks:
                t = threading.Thread(target=update_chunk, args=(chunk,))
                t.start()
                time.sleep(sleep)
    
    def chunks(self):
        return Chunk.objects.filter(doc=self).order_by('chunk_index')
    
    def progress(self):
        total = self.chunks().count()
        complete = self.chunks().filter(summary_array__isnull=False).count()
        return complete, total, complete == total
    
