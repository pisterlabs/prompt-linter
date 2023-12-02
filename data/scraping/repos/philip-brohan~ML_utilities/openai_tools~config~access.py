# Utility function to get the API key

import os
import configparser

# Location of key file on disc (system dependent)
keyfile = "%s/.openai/config" % os.getenv('HOME')

# This file should have contents like: (without the '#')
# [organisation]
# API_key = <Your API key from openai website>

def get_key():
    cfg = configparser.ConfigParser()
    cfg.read(keyfile)
    return(cfg.get('organisation', 'API_key'))
