from odoo import models, api, fields
import openai

class MindMapDiagram(models.Model):
    _name = 'mindmap.diagram'
    _rec_name = 'name'

    name = fields.Char(string='Name')
    uml_diagram = fields.Binary(string='Mind Map',widget='image')
    uml_code = f"""@startmindmap
                    * Debian
                    ** Ubuntu
                    *** Linux Mint
                    *** Kubuntu
                    *** Lubuntu
                    *** KDE Neon
                    ** LMDE
                    ** SolydXK
                    ** SteamOS
                    ** Raspbian with a very long name
                    *** <s>Raspmbc</s> => OSMC
                    *** <s>Raspyfi</s> => Volumio
                @endmindmap"""
 
class PromptDynamicGPT(models.Model):
    _name = 'dynamic.prompt'
    _rec_name = 'record_name'

    select_type = fields.Selection([('requirements','Requirements'),('roles','Roles and Activities')], string='Select Intent', default='requirements', required='1')
    prompt_input = fields.Char(string='Enter a Prompt')
    description = fields.Char(string='Project Description')
    role = fields.Char(string='User Role')
    content = fields.Char(string='Content', default='List Out The Roles And Qualities')
    api_key = fields.Char(string='Api Key',required=True)
    model = fields.Selection([('text-davinci-003','text-davinci-003'),('gpt-3.5-turbo','gpt-3.5-turbo')],string='Select Model',required=True, default='text-davinci-003')
    temperature = fields.Float(string='Temperature', default=1.25,required=True)
    max_tokens = fields.Integer(string='Enter Max Tokens', default=100,required=True)
    top_p = fields.Integer(string='Top P', default=1,required=True)
    frequency_penalty = fields.Integer(string='Frequency Penalty', default=0,required=True)
    presence_penalty = fields.Integer(string='Presence Penalty', default=0,required=True)
    record_name = fields.Char(string='Rec Name', compute='generate_rec')

    def generate_rec(self):
        for i in self:
            if i.description:
                i.record_name = i.description
            else:
                i.record_name = i.content
                
