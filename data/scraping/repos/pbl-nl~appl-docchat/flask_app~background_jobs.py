import json
from os import path, remove
from time import sleep
import datetime
from sqlalchemy.sql import func, or_
from concurrent.futures import ThreadPoolExecutor
from flask import current_app

from ingest.ingester import Ingester
from flask_app.models import db, Job, DocSet, DocSetFile

from langchain.vectorstores import Chroma

'''
status_system:  Pending
                Running
                Done | Error
status:   1  Pending                 when a job is inserted, the status is Pending
          2  -job-type-dependent-    job_type == 'Ingest'        status is Chunking
                                     job_type == 'Delete file'   status is Deleting
                                     job_type == '...'           status is ...
          3  Done | Error            when a job is finished, the status is Done or Error
'''
class BackgroundJobs:
    bg_job_app = None
    bg_job_executor = None

    def init_app(self, app):
        self.bg_job_app = app
        self.bg_job_executor = ThreadPoolExecutor(max_workers=1)
        self.clear_jobs(True)


    def clear_jobs(self, on_init=False):
        if on_init:
            jobs = Job.query.filter(Job.status_system != 'Pending').all()
        else:
            current_time = datetime.datetime.utcnow()
            two_hours_ago = current_time - datetime.timedelta(hours=2)
            jobs = Job.query.filter(Job.dt_last < two_hours_ago).all()
        #jobs.delete()
        #db.session.commit()


    def new_job(self, job_type, bind_to_id, **kwargs):
        job = Job()
        job.job_type = job_type
        job.status_system = 'Pending'

        if job.job_type == 'Ingest':
            job.status = 'Chunking pending'
        elif job.job_type == 'Delete file':
            job.status = 'Delete pending'

        job.bind_to_id = bind_to_id
        job.status_msg = ''
        job.job_parms = json.dumps(kwargs)
        db.session.add(job)
        db.session.commit()
        self.bg_job_executor.submit(self.background_job, job.id)

    def background_job(self, job_id):
        with self.bg_job_app.app_context():
            try:
                job = Job.query.filter(Job.id == job_id).first()
                if job:
                    print('Start job', job, flush=True)
                    try:

                        job.status_system = 'Running'
                        if job.job_type == 'Ingest':
                            job.status = 'Chunking'
                        elif job.job_type == 'Delete file':
                            job.status = 'Deleting'

                        db.session.commit()
                        parms = json.loads(job.job_parms)

                        if job.job_type == 'Ingest':
                            docsetfile = DocSetFile.query.filter(DocSetFile.id == job.bind_to_id).first()
                            if docsetfile:
                                docsetfiles = DocSetFile.query.filter(DocSetFile.docset_id == docsetfile.docset_id, DocSetFile.no >= 1).all()
                                no = 1
                                if docsetfiles:
                                    for docsetfile_ in docsetfiles:
                                        if docsetfile_.no >= no:
                                            no = docsetfile_.no + 1
                                docsetfile.no = no
                                db.session.commit()
                                ok, msg = self.job_ingest(parms['docset_id'], parms['filename'], docsetfile.no)
                            else:
                                ok, msg = False, 'The file has been deleted.'
                        elif job.job_type == 'Delete file':
                            ok, msg = self.job_delete_file(job.bind_to_id)
                        
                        job.status = 'Done' if ok else 'Error'
                        job.status_system = 'Done' if ok else 'Error'
                        job.status_msg = msg
                        db.session.commit()
                    except Exception as e:
                        job.status = 'Error'
                        job.status_system = 'Error'
                        job.status_msg = type(e).__name__ + ' ' + str(e)
                        db.session.commit()
                    print('End job', job, flush=True)
            except Exception as e:
                print('Severe error in background job (background jobs stopped running):', e, flush=True)


    def job_ingest(self, docset_id, filename, file_no):
        try:
            docset = DocSet.query.filter(DocSet.id == docset_id).first()
            if docset:
                #print('background_job: ingest(docset, filename, file_no)', docset, filename, file_no)
                ingester = Ingester(
                    docset.get_collection_name(), 
                    path.join(docset.get_doc_path(), filename),
                    docset.create_vectordb_name(), 
                    embeddings_provider=docset.embeddings_provider, 
                    embeddings_model=docset.embeddings_model, 
                    text_splitter_method=docset.text_splitter_method, 
                    vecdb_type=docset.vecdb_type,
                    chunk_size=docset.chunk_size,
                    chunk_overlap=docset.chunk_overlap,
                    file_no=file_no
                )
                ingester.ingest()
                ok, msg = True, 'Ingested'
            else:
                ok, msg = False, 'Document set does not exists (anymore).'
        except Exception as e:
            ok, msg = False, type(e).__name__ + ' ' + str(e)
        return ok, msg

    def job_delete_file(self, file_id):
        obj = DocSetFile.query.get(file_id)
        if obj:
            docset = DocSet.query.get(obj.docset_id)
            if docset:
                vectordb_folder = docset.create_vectordb_name()
                vector_store = Chroma(
                                collection_name=docset.get_collection_name(),
                                persist_directory=vectordb_folder,
                            )
                sources = vector_store.get()
                i, ids_to_delete = 0, []
                for vec_id in sources['ids']:
                    if sources['metadatas'][i]['file_no'] == obj.no:
                        ids_to_delete.append(vec_id)
                    i += 1
                vector_store.delete(ids=ids_to_delete)
                vector_store.persist()
                full_filename = path.join(docset.get_doc_path(), obj.filename)
                if path.exists(full_filename):
                    remove(full_filename)
                ok, msg = True, 'The file \'' + obj.filename + '\' is deleted.'
                # Delete all related jobs
                Job.query.filter(Job.bind_to_id == obj.id, or_(Job.job_type == 'Ingest', Job.job_type == 'Delete file')).delete()
                DocSetFile.query.filter(DocSetFile.id == obj.id).delete()
                db.session.commit()
            else:
                ok, msg = False, 'The document set is no longer in the database.'
        else:
            ok, msg = False, 'The file is no longer in the database.'
        return ok, msg
    
background_jobs = BackgroundJobs()