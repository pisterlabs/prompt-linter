# -*- coding: utf-8 -*-

from odoo import models, fields, api, _
from odoo.exceptions import ValidationError, UserError


class OdoogptOpenaiMixin(models.AbstractModel):
    _name = 'odoogpt.openai.mixin'
    _description = 'OdooGPT OpenAI Mixin'
    _rec_name = 'openai_id'

    _sql_constraints = [
        ('UNIQUE_OPENAI_ID', 'UNIQUE(openai_id)', _('The OpenAI Id must be unique!'))
    ]


    # OpenAI record type, field `object`
    REC_TYPES = []


    openai_id = fields.Char(
        string='OpenAI Id',
        required=False,
        default=False,
        index=True,
    )


    # UTILS
    def _get_as_dict(self, domain=[]):
        """Get all records in a dict format (openai_id: record)"""
        records = self.search(domain)

        return {rec.openai_id: rec for rec in records}


    # REFRESH FROM API
    @api.model
    def _refresh_from_api(self):
        return []

    @property
    def _refresh_from_api_fields(self):
        return []

    @api.model
    def refresh_from_api(self, format='model'):
        """Refresh records stored in database from OpenAI apis"""
        # TODO: Check what to do with unlinking of inexistent records. Maybe we
        #       can keep them.
        # OdoogptOpenaiUtils = self.env['odoogpt.openai.utils']

        # Check and get OpenAI Api Key
        # api_key = OdoogptOpenaiUtils._odoogpt_check_api_key(raise_err=True)

        # Get records from OpenAI
        try:
            openai_records = self._refresh_from_api()
        except Exception as ex:
            raise UserError(ex)

        if not openai_records and not len(openai_records):
            raise ValidationError(_('No records found from OpenAI api'))

        # Store/update records in our database
        odoogpt_records = self._get_as_dict()
        write_fields = self._refresh_from_api_fields

        for openai_record in openai_records:
            if openai_record.get('object', '') not in self.REC_TYPES:
                continue

            odoogpt_record = odoogpt_records.get(openai_record['id'])
            if odoogpt_record:
                odoogpt_record.write(
                    {key: openai_record.get(key, odoogpt_record[key]) for key in write_fields}
                )
            elif openai_record.get('id'):
                odoogpt_record = self.create({
                    'openai_id': openai_record.get('id'),
                    **{key: openai_record.get(key) for key in write_fields}
                })

                odoogpt_records[odoogpt_record.openai_id] = odoogpt_record

        # Return
        if format == 'dict':
            return odoogpt_records
        else:   # format == 'model' or anything else
            ret = self
            for rec in odoogpt_records.values():
                ret += rec