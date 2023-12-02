from flask_restful import Resource, reqparse
from flask_limiter.util import get_remote_address
from flask import session, jsonify, request
import logging
import os
import openai
from flask_limiter import Limiter

#CHAT for economics that triggers datapipeline from mongodb to client side for plot visualization