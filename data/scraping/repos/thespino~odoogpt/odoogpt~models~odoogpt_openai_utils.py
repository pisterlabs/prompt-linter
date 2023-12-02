# -*- coding: utf-8 -*-

from odoo import models, _
from odoo.exceptions import ValidationError
import openai


class OdoogptOpenaiUtils(models.AbstractModel):
    _name = 'odoogpt.openai.utils'
    _description = 'OdooGPT OpenAI Utils'


    # UTILS ==============================================================================

    def _odoogpt_check_api_key(self, raise_err=True):
        """Check api key existance in user/company"""
        if self.env.user.odoogpt_openai_api_key:
            return self.env.user.odoogpt_openai_api_key
        if raise_err:
            raise ValidationError(_('OpenAI Api Token not set in settings!'))
        else:
            return False

    def _odoogpt_get_parameters(self, PARAMETERS):
        """Get params to pass to openai library to make a successful request"""
        params = {}
        for PARAM in PARAMETERS:
            params[PARAM[1]] = self.env.user[PARAM[0]]

        return params


    # MODELS =============================================================================

    def _models_list__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def models_list(self, **kwargs):
        """List available Models. Ref. https://platform.openai.com/docs/api-reference/models/list"""
        response = openai.Model.list(
            **{
                **self._models_list__get_parameters(),
                **kwargs,
            }
        )

        return response['data']


    # COMPLETITION =======================================================================

    def _completition_create__get_parameters(self):
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
            ('odoogpt_openai_model', 'model'),
            ('odoogpt_openai_max_tokens', 'max_tokens'),
            ('odoogpt_openai_temperature', 'temperature'),
        ])

    def completition_create(self, prompt, **kwargs):
        """Create a Completition. Ref. https://platform.openai.com/docs/api-reference/completions"""
        response = openai.Completion.create(
            prompt=prompt,
            **{
                **self._completition_create__get_parameters(),
                **kwargs,
            }
        )

        # Log
        self.env['odoogpt.openai.log'].sudo().create({
            'type': 'completition',
            'raw_request': prompt,
            'parsed_request': prompt,
            'raw_response': str(response),
            'parsed_response': response['choices'][0]['text'],
        })

        return response['choices'][0]['text']


    # CHAT COMPLETION ====================================================================

    def _chat_completion_create__get_parameters(self):
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
            ('odoogpt_openai_model', 'model'),
            ('odoogpt_openai_max_tokens', 'max_tokens'),
            ('odoogpt_openai_temperature', 'temperature'),
        ])

    def chat_completion_create(self, messages, **kwargs):
        """Create a Chat Completition. Ref. https://platform.openai.com/docs/api-reference/chat/create"""
        response = openai.ChatCompletion.create(
            messages=messages,
            **{
                **self._completition_create__get_parameters(),
                **kwargs,
            }
        )

        # Log
        self.env['odoogpt.openai.log'].sudo().create({
            'type': 'chat-completion',
            'raw_request': str(messages),
            'parsed_request': str(messages),
            'raw_response': str(response),
            'parsed_response': response['choices'][0]['message']['content'],
        })

        return response['choices'][0]['message']['content']


    # FILES ==============================================================================

    def _files_list__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def files_list(self, **kwargs):
        """List Files. Ref. https://platform.openai.com/docs/api-reference/files/list"""
        response = openai.File.list(
            **{
                **self._files_list__get_parameters(),
                **kwargs,
            }
        )

        return response['data']


    def _files_create__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def files_create(self, file, purpose='fine-tune', **kwargs):
        """Upload/Create File. Ref. https://platform.openai.com/docs/api-reference/files/upload"""
        response = openai.File.create(
            file=file,
            purpose=purpose,
            **{
                **self._files_create__get_parameters(),
                **kwargs,
            }
        )

        return response


    def _files_delete__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def files_delete(self, file_id, **kwargs):
        """Delete File. Ref. https://platform.openai.com/docs/api-reference/files/delete"""
        response = openai.File.delete(
            sid=file_id,
            **{
                **self._files_delete__get_parameters(),
                **kwargs,
            }
        )

        return response


    def _files_download__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def files_download(self, file_id, **kwargs):
        """Download File. Ref. https://platform.openai.com/docs/api-reference/files/retrieve-content"""
        response = openai.File.download(
            id=file_id,
            **{
                **self._files_download__get_parameters(),
                **kwargs,
            }
        )

        return response


    # FINE TUNES =========================================================================

    def _fine_tunes_list__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def fine_tunes_list(self, **kwargs):
        """List Fine Tunes. Ref. https://platform.openai.com/docs/api-reference/fine-tunes/list"""
        response = openai.FineTune.list(
            **{
                **self._fine_tunes_list__get_parameters(),
                **kwargs,
            }
        )

        return response['data']


    def _fine_tunes_create__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def fine_tunes_create(self, training_file, **kwargs):
        """Create Fine Tune. Ref. https://platform.openai.com/docs/api-reference/fine-tunes/create"""
        response = openai.FineTune.create(
            training_file=training_file,
            **{
                **self._fine_tunes_create__get_parameters(),
                **kwargs,
            }
        )

        return response


    def _fine_tunes_cancel__get_parameters(self):
        """Get params to pass to openai library to make a successful request"""
        return self._odoogpt_get_parameters([
            # Odoo name, OpenAI name
            ('odoogpt_openai_api_key', 'api_key'),
        ])

    def fine_tunes_cancel(self, fine_tune_id, **kwargs):
        """Cancel Fine Tune job. Ref. https://platform.openai.com/docs/api-reference/fine-tunes/cancel"""
        response = openai.FineTune.cancel(
            id=fine_tune_id,
            **{
                **self._fine_tunes_cancel__get_parameters(),
                **kwargs,
            }
        )

        return response
