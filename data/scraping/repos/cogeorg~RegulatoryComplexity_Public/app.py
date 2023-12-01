#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
import os
import sys
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from apps import welcome
from apps import words
#from apps import sentences
from apps import coherence


# add project directory to the sys.path
project_home = os.getcwd()
if project_home not in sys.path:
    sys.path.append(project_home)


# initiate the combined application
application = Flask(__name__)
application.config.update(
    TEMPLATES_AUTO_RELOAD = True
)

application.wsgi_app = DispatcherMiddleware(welcome, {
    '/words':           words,
#    '/sentenceparts':   sentences,
    '/coherence':       coherence
})

# run the application
if __name__ == "__main__":
    application.run(debug = True)
