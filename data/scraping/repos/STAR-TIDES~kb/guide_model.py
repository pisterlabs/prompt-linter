''' star_tides.services.mongo.models.user_model
'''
from mongoengine import (
    Document,
    StringField,
    UUIDField,
    ListField,
    EmbeddedDocumentField
)

from star_tides.services.databases.mongo.models.guidance_model import Guidance
from star_tides.services.databases.mongo.models.project_model import (
    ProjectModel
)
from star_tides.services.databases.mongo.models.contact_model import (
    ContactModel
)


class Guide(Document):
    ''' User model
    '''
    uuid = UUIDField(binary=False, required=False)
    name = StringField(required=True)
    focuses = ListField(StringField, required=True) # @LJR do we have a
                                                    # focus model?
    summary = StringField(required=False)
    related_projects = ListField(ProjectModel, required=False, default=[])
    relevant_contacts = ListField(ContactModel, required=False, default=[])
    guidance = ListField(
        EmbeddedDocumentField(
            Guidance,
            required=True
        ),
        required=False,
        default=[]
    )
