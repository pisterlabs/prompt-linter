#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : etc.py
# Create date : 2019-03-20 19:53
# Modified date : 2019-03-22 15:13
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch
from openai_gpt2.etc import config as gpt2_config
from openai_gpt.etc import config as gpt_config

config = {}

config["gpt2_config"] = gpt2_config
config["gpt_config"] = gpt_config
