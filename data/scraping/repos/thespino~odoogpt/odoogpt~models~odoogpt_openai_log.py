# -*- coding: utf-8 -*-

from odoo import models, fields, _


class OdoogptOpenaiLog(models.Model):
    _name = 'odoogpt.openai.log'
    _description = 'OdooGPT OpenAI Log'
    _rec_name = 'type'


    type = fields.Selection(
        string='Request Type',
        help="""Type of request""",
        selection=[
            ('completition', 'Completition'),
            ('chat-completion', 'Chat Completion'),
        ],
        required=True,
    )


    raw_request = fields.Text(
        string='Raw Request',
        help="""Request, as-is, sent to OpenAI Api""",
        required=True,
    )

    raw_response = fields.Text(
        string='Raw Response',
        help="""Response, as-is, received from OpenAI Api""",
        required=True,
    )


    parsed_request = fields.Text(
        string='Parsed Request',
        help="""Parsed Request, to be human readable""",
        required=False,
    )

    parsed_response = fields.Text(
        string='Parsed Response',
        help="""Parsed Response, to be human readable""",
        required=False,
    )
