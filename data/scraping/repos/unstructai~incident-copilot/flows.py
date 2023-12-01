"""
.. Modified by Kishore Jalleda
.. full list of modifications at https://github.com/unstructai
.. copyright: (c) 2022 Kishore Jalleda
.. author:: Kishore Jalleda <kjalleda@gmail.com>
"""
import logging

from datetime import datetime
from typing import Any, List, Tuple, Optional

from dispatch.conference import service as conference_service
from dispatch.conference.models import ConferenceCreate
from dispatch.conversation import service as conversation_service
from dispatch.conversation.models import ConversationCreate
from dispatch.database.core import SessionLocal, resolve_attr
from dispatch.decorators import background_task
from dispatch.document import service as document_service
from dispatch.document.models import DocumentCreate
from dispatch.enums import DocumentResourceTypes
from dispatch.enums import Visibility, EventType
from dispatch.event import service as event_service
from dispatch.group import service as group_service
from dispatch.group.models import GroupCreate
from dispatch.incident import service as incident_service
from dispatch.incident.models import IncidentRead
from dispatch.incident.type import service as incident_type_service
from dispatch.individual import service as individual_service
from dispatch.messaging.strings import (
    INCIDENT_INVESTIGATION_DOCUMENT_DESCRIPTION,
    INCIDENT_INVESTIGATION_SHEET_DESCRIPTION,
    INCIDENT_CONVERSATION_DESCRIPTION,
)
from dispatch.participant import flows as participant_flows
from dispatch.participant import service as participant_service
from dispatch.participant.models import Participant
from dispatch.participant_role import flows as participant_role_flows
from dispatch.participant_role.models import ParticipantRoleType
from dispatch.plugin import service as plugin_service
from dispatch.report.enums import ReportTypes
from dispatch.report.messaging import send_incident_report_reminder
from dispatch.service import service as service_service
from dispatch.storage import service as storage_service
from dispatch.storage.models import StorageCreate
from dispatch.task.enums import TaskStatus, TaskType
from dispatch.task import service as task_service
from dispatch.ticket import service as ticket_service
from dispatch.ticket.models import TicketCreate
from dispatch.postmortem import service as postmortem_service
from dispatch.postmortem.models import PostmortemCreate

from timesketch_api_client import config
from dispatch.config import (
    STORYCURVE_UI_URL,
    DISPATCH_UI_URL,
    UNALERT_UI_URL,
    UNSTATUS_UI_URL,
)

from dispatch.ticket import flows as ticket_flows
from dispatch.storage import flows as storage_flows
from dispatch.group import flows as group_flows
from dispatch.conversation import flows as conversation_flows
from dispatch.AI.prompt import config as prompt_config
from dispatch.AI.prompt import service as prompt_service


from dispatch.nlp import extract_tags

from sqlalchemy.orm import Session


from .messaging import (
    get_suggested_document_items,
    send_incident_closed_information_review_reminder,
    send_incident_commander_readded_notification,
    send_incident_created_notifications,
    send_incident_management_help_tips_message,
    send_incident_new_role_assigned_notification,
    send_incident_open_tasks_ephemeral_message,
    send_incident_participant_announcement_message,
    send_incident_rating_feedback_message,
    send_incident_review_document_notification,
    send_incident_suggested_reading_messages,
    send_incident_update_notifications,
    send_incident_welcome_participant_messages,
    send_postmortem_conference_notification,
)
from .models import Incident, IncidentStatus


log = logging.getLogger(__name__)


def get_incident_participants(incident: Incident, db_session: SessionLocal):
    """Get additional incident participants based on priority, type, and description."""
    individual_contacts = []
    team_contacts = []

    if incident.visibility == Visibility.open:
        plugin = plugin_service.get_active_instance(
            db_session=db_session, project_id=incident.project.id, plugin_type="participant"
        )
        if plugin:
            individual_contacts, team_contacts = plugin.instance.get(
                class_instance=incident,
                project_id=incident.project.id,
                db_session=db_session,
            )

    return individual_contacts, team_contacts


def reactivate_incident_participants(incident: Incident, db_session: SessionLocal):
    """Reactivates all incident participants."""
    for participant in incident.participants:
        try:
            incident_add_or_reactivate_participant_flow(
                participant.individual.email, incident.id, db_session=db_session
            )
        except Exception as e:
            # don't fail to reactivate all participants if one fails
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Unable to reactivate participant with email {participant.individual.email}",
                incident_id=incident.id,
                type=EventType.participant_updated,
            )
            log.exception(e)

    event_service.log_incident_event(
        db_session=db_session,
        source="unStruct Core App",
        description="Incident participants reactivated",
        incident_id=incident.id,
        type=EventType.participant_updated,
    )


def inactivate_incident_participants(incident: Incident, db_session: SessionLocal):
    """Inactivates all incident participants."""
    for participant in incident.participants:
        try:
            participant_flows.inactivate_participant(
                participant.individual.email, incident, db_session
            )
        except Exception as e:
            # don't fail to inactivate all participants if one fails
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Unable to inactivate participant with email {participant.individual.email}",
                incident_id=incident.id,
                type=EventType.participant_updated,
            )
            log.exception(e)

    event_service.log_incident_event(
        db_session=db_session,
        source="unStruct Core App",
        description="Incident participants inactivated",
        incident_id=incident.id,
        type=EventType.participant_updated,
    )


def create_incident_ticket(incident: Incident, db_session: SessionLocal, parent_id: str = None):
    """Create an external ticket for tracking."""
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="ticket"
    )
    if not plugin:
        log.warning("Incident ticket not created. No ticket plugin enabled.")
        return

    title = incident.title
    if incident.visibility == Visibility.restricted:
        title = incident.incident_type.name

    incident_type_plugin_metadata = incident_type_service.get_by_name_or_raise(
        db_session=db_session,
        project_id=incident.project.id,
        incident_type_in=incident.incident_type,
    ).get_meta(plugin.plugin.slug)

    ticket = plugin.instance.create(
        incident.id,
        title,
        incident.commander.individual.email,
        incident.reporter.individual.email,
        incident_type_plugin_metadata,
        parent_id=parent_id,
        db_session=db_session,
    )
    ticket.update({"resource_type": plugin.plugin.slug})

    # TODO - move back and refactor?
    if ticket:
        incident.ticket = ticket_service.create(
            db_session=db_session, ticket_in=TicketCreate(**ticket)
        )

        # we set the incident name based on Jira or SNOW
        if ticket.get("resource_id"):
            if ticket.get("number"):
                incident.name = ticket["number"]
            else:
                incident.name = ticket["resource_id"]
        else:
            incident.name = f"TBD-{incident.id}"

    # Commit all the changes to DB
    db_session.add(incident)
    db_session.commit()

    event_service.log_incident_event(
        db_session=db_session,
        source=plugin.plugin.title,
        description="Ticket created",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )

    return ticket


def update_external_incident_ticket(
    incident_id: int,
    db_session: SessionLocal,
):
    """Update external incident ticket."""
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="ticket"
    )
    if not plugin:
        log.warning("External ticket not updated. No ticket plugin enabled.")
        return

    title = incident.title
    description = incident.description
    if incident.visibility == Visibility.restricted:
        title = description = incident.incident_type.name

    incident_type_plugin_metadata = incident_type_service.get_by_name_or_raise(
        db_session=db_session,
        project_id=incident.project.id,
        incident_type_in=incident.incident_type,
    ).get_meta(plugin.plugin.slug)

    total_cost = 0
    if incident.total_cost:
        total_cost = incident.total_cost

    plugin.instance.update(
        incident.ticket.resource_id,
        title,
        description,
        incident.incident_type.name,
        incident.incident_severity.name,
        incident.incident_priority.name,
        incident.status.lower(),
        incident.commander.individual.email,
        incident.reporter.individual.email,
        resolve_attr(incident, "conversation.weblink"),
        resolve_attr(incident, "incident_document.weblink"),
        resolve_attr(incident, "storage.weblink"),
        resolve_attr(incident, "conference.weblink"),
        total_cost,
        incident_type_plugin_metadata=incident_type_plugin_metadata,
    )

    event_service.log_incident_event(
        db_session=db_session,
        source=plugin.plugin.title,
        description=f"Ticket updated. Status: {incident.status}",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )


def create_participant_groups(
    incident: Incident,
    direct_participants: List[Any],
    indirect_participants: List[Any],
    db_session: SessionLocal,
):
    """Create external participant groups."""
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="participant-group"
    )

    group_name = f"{incident.name}"
    notifications_group_name = f"{group_name}-notifications"

    direct_participant_emails = [x.email for x, _ in direct_participants]
    tactical_group = plugin.instance.create(
        group_name, direct_participant_emails
    )  # add participants to core group

    indirect_participant_emails = [x.email for x in indirect_participants]
    indirect_participant_emails.append(
        tactical_group["email"]
    )  # add all those already in the tactical group
    notifications_group = plugin.instance.create(
        notifications_group_name, indirect_participant_emails
    )

    tactical_group.update(
        {
            "resource_type": f"{plugin.plugin.slug}-tactical-group",
            "resource_id": tactical_group["id"],
        }
    )
    notifications_group.update(
        {
            "resource_type": f"{plugin.plugin.slug}-notifications-group",
            "resource_id": notifications_group["id"],
        }
    )

    event_service.log_incident_event(
        db_session=db_session,
        source=plugin.plugin.title,
        description="Tactical and notifications groups created",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )

    return tactical_group, notifications_group


def create_conference(incident: Incident, participants: List[str], db_session: SessionLocal):
    """Create external conference room."""
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conference"
    )
    conference = plugin.instance.create(incident.name, participants=participants)

    conference.update({"resource_type": plugin.plugin.slug, "resource_id": conference["id"]})

    event_service.log_incident_event(
        db_session=db_session,
        source=plugin.plugin.title,
        description="Incident conference created",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )

    return conference


def create_incident_storage(
    incident: Incident, participant_group_emails: List[str], db_session: SessionLocal
):
    """Create an external file store for incident storage."""
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="storage"
    )
    storage_root_id = plugin.configuration.root_id
    storage = plugin.instance.create_file(storage_root_id, incident.name, participant_group_emails)
    storage.update({"resource_type": plugin.plugin.slug, "resource_id": storage["id"]})
    return storage


def create_incident_documents(incident: Incident, db_session: SessionLocal):
    """Create incident documents."""
    incident_documents = []

    if not incident.storage:
        return incident_documents

    # we get the storage plugin
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="storage"
    )

    if plugin:
        # we create the investigation document
        incident_document_name = f"{incident.name} - Incident Document"
        incident_document_description = INCIDENT_INVESTIGATION_DOCUMENT_DESCRIPTION

        if incident.incident_type.incident_template_document:
            incident_document_description = (
                incident.incident_type.incident_template_document.description
            )
            document = plugin.instance.copy_file(
                incident.storage.resource_id,
                incident.incident_type.incident_template_document.resource_id,
                incident_document_name,
            )
            plugin.instance.move_file(incident.storage.resource_id, document["id"])
        else:
            # we create a blank document if no template is defined
            document = plugin.instance.create_file(
                incident.storage.resource_id, incident_document_name, file_type="document"
            )

        document.update(
            {
                "name": incident_document_name,
                "description": incident_document_description,
                "resource_type": DocumentResourceTypes.incident,
                "resource_id": document["id"],
            }
        )

        incident_documents.append(document)

        event_service.log_incident_event(
            db_session=db_session,
            source=plugin.plugin.title,
            description="Incident document created",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )

        # we create the investigation sheet
        incident_sheet_name = f"{incident.name} - Incident Tracking Sheet"
        incident_sheet_description = INCIDENT_INVESTIGATION_SHEET_DESCRIPTION

        if incident.incident_type.tracking_template_document:
            incident_sheet_description = (
                incident.incident_type.tracking_template_document.description
            )
            sheet = plugin.instance.copy_file(
                incident.storage.resource_id,
                incident.incident_type.tracking_template_document.resource_id,
                incident_sheet_name,
            )
            plugin.instance.move_file(incident.storage.resource_id, sheet["id"])
        else:
            # we create a blank sheet if no template is defined
            sheet = plugin.instance.create_file(
                incident.storage.resource_id, incident_sheet_name, file_type="sheet"
            )

        if sheet:
            sheet.update(
                {
                    "name": incident_sheet_name,
                    "description": incident_sheet_description,
                    "resource_type": DocumentResourceTypes.tracking,
                    "resource_id": sheet["id"],
                }
            )

            incident_documents.append(sheet)

        event_service.log_incident_event(
            db_session=db_session,
            source=plugin.plugin.title,
            description="Incident sheet created",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )

        # we create folders to store logs and screengrabs
        plugin.instance.create_file(incident.storage.resource_id, "logs")
        plugin.instance.create_file(incident.storage.resource_id, "screengrabs")

    return incident_documents


def create_post_incident_review_document(incident: Incident, db_session: SessionLocal):
    """Create post-incident review document."""
    # we get the storage plugin
    storage_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="storage"
    )
    if not storage_plugin:
        log.warning("Post-incident review document not created. No storage plugin enabled.")
        return

    # we create a copy of the incident review document template
    # and we move it to the incident storage
    incident_review_document_name = f"{incident.name} - Post-Incident Review Document"

    # incident review document is optional
    if not incident.incident_type.review_template_document:
        log.warning("No template for post-incident review document has been specified.")
        return

    # we create the document
    incident_review_document = storage_plugin.instance.copy_file(
        folder_id=incident.storage.resource_id,
        file_id=incident.incident_type.review_template_document.resource_id,
        name=incident_review_document_name,
    )

    incident_review_document.update(
        {
            "name": incident_review_document_name,
            "description": incident.incident_type.review_template_document.description,
            "resource_type": DocumentResourceTypes.review,
        }
    )

    # we move the document to the storage
    storage_plugin.instance.move_file(
        new_folder_id=incident.storage.resource_id,
        file_id=incident_review_document["id"],
    )

    event_service.log_incident_event(
        db_session=db_session,
        source=storage_plugin.plugin.title,
        description="Post-incident review document added to storage",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )

    # we add the document to the incident
    document_in = DocumentCreate(
        name=incident_review_document["name"],
        description=incident_review_document["description"],
        resource_id=incident_review_document["id"],
        resource_type=incident_review_document["resource_type"],
        project=incident.project,
        weblink=incident_review_document["weblink"],
    )

    incident_review_document = document_service.create(
        db_session=db_session, document_in=document_in
    )
    incident.documents.append(incident_review_document)
    incident.incident_review_document_id = incident_review_document.id

    event_service.log_incident_event(
        db_session=db_session,
        source="unStruct Core App",
        description="Post-incident review document added to incident",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )

    # we update the post-incident review document
    update_document(incident.incident_review_document.resource_id, incident, db_session)

    # we create a story in... storycurve and share the story.
    if incident.incident_priority.create_incident_postmortem:
        result, exception = create_postmortem_story(incident_id=incident.id, db_session=db_session)
        if result:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="A postmortem document has automatically been generated for this incident (based on configuration for incident priority). A meeting will be scheduled to discuss the incident and the post-incident review document and shared with all participants",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
        else:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Error in automatic creation a postmortem document for this incident (based on configuration for incident priority). Please use the create postmortem link to create a postmortem document. Error: {exception}",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
    else:
        event_service.log_incident_event(
            db_session=db_session,
            source="unStruct Core App",
            description="Skipped automatically creating a postmortem document for this incident (based on configuration for incident priority). Please use the create postmortem link to create a postmortem document.",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )

    # We update the statuspage with the post-incident review document. TODO.
    # We update the statuspage
    statuspage_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="statuspage"
    )

    # only update if the incident is linked to a statuspage
    if (
        statuspage_plugin
        and incident.statuspage_id
        and incident.incident_priority.auto_post_to_status_page
    ):
        try:
            statuspage_plugin.instance.resolve_incident(
                id=incident.statuspage_id,
                description="A post-incident review document has been created for this incident. A meeting will be scheduled to discuss the incident and the post-incident review document.",
            )
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Incident status updated in statuspage",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
        except Exception as e:
            log.warning(f"Failed to update incident in the status page {e}")

    db_session.add(incident)
    db_session.commit()


def update_document(document_resource_id: str, incident: Incident, db_session: SessionLocal):
    """Updates an existing document."""
    # we get the document plugin
    document_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="document"
    )

    if not document_plugin:
        log.warning("Document not updated. No document plugin enabled.")
        return

    # Construct time in a str() object - KJ
    timeline = str()
    events = incident.events
    for e in events:
        timeline += f"{e.started_at.replace(second=0, microsecond=0)}        {e.description}\n\n"

    document_plugin.instance.update(
        document_resource_id,
        commander_fullname=incident.commander.individual.name,
        conference_challenge=resolve_attr(incident, "conference.challenge"),
        conference_weblink=resolve_attr(incident, "conference.weblink"),
        conversation_weblink=resolve_attr(incident, "conversation.weblink"),
        description=incident.description,
        document_weblink=resolve_attr(incident, "incident_document.weblink"),
        name=incident.name,
        priority=incident.incident_priority.name,
        resolution=incident.resolution,
        severity=incident.incident_severity.name,
        status=incident.status,
        storage_weblink=resolve_attr(incident, "storage.weblink"),
        ticket_weblink=resolve_attr(incident, "ticket.weblink"),
        title=incident.title,
        type=incident.incident_type.name,
        timeline=timeline,
        reported_at_time=incident.reported_at.strftime("%m/%d/%Y %H:%M:%S"),
        stable_at_time=incident.stable_at.strftime("%m/%d/%Y %H:%M:%S"),
    )


def create_postmortem_story(
    incident_id: int, db_session: SessionLocal
) -> Tuple[bool, Optional[Exception]]:
    """Creates a postmortem story."""

    # Fetch the incident.
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    # We create a postmortem in the DB.
    postmortem_in = PostmortemCreate(
        incident_id=incident_id,
        title=f"{incident.name} - Postmortem",
        project=incident.project,
    )

    postmortem = postmortem_service.create(db_session=db_session, postmortem_in=postmortem_in)

    # Get the storycurve client
    try:
        sc_client = config.get_client()
        sketch = sc_client.get_sketch(incident.storycurve_sketch_id)

        # KJ TODO. Think of a better way to display critical events.
        important_events_view = sketch.create_view(
            name="Timeline of Critical Events",
            query_string='tag: ("Key Decision Point" OR Hypothesis OR Risk)',
        )
        all_events_view = sketch.create_view(
            name="Full Timeline",
            query_string="*",
        )

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story = sketch.create_story(
            f"[Postmortem] [{incident.name}] {incident.title} -- {current_time}"
        )
        story.add_text(
            f"This report was automatically created by the UnStruct AI Bot at {current_time}"
        )

        # Create guidelines from openAI
        plugin = plugin_service.get_preferred_active_instance(
            db_session=db_session,
            project_id=incident.project.id,
            plugin_type="artificial-intelligence",
        )

        story.add_text("# Postmortem Guidelines")
        if plugin:
            prompt = "How to conduct a postmortem?"
            r_text = plugin.instance.ask(prompt=prompt)

            story.add_text(r_text)
        else:
            log.warning("No AI plugin enabled. Unable to generate guidelines.")
            story.add_text("No AI plugin enabled. Unable to generate guidelines.")

        incident_participants_str = ""
        for participant in incident.participants:
            roles = ""
            for pr in participant.participant_roles:
                roles += f"{pr.role}, "
            incident_participants_str += f"{participant.individual.name} ({participant.individual.email}) (Role(s): {roles}) \n\n"

        custom_fields_str = ""
        # Check if custom fields exist. TODO. Copy over from type defs
        for custom_field in incident.custom_fields if incident.custom_fields else []:
            custom_fields_str += f'{custom_field["name"]}: {custom_field["value"]}\n\n'

        story.add_text("# Incident Metadata")
        story.add_text(
            f"**Priority**:\n {incident.incident_priority.name}\n\n"
            f"**Reported At**:\n {incident.reported_at}\n\n"
            f"**Stable At**:\n {incident.stable_at}\n\n"
            f"**Resolution**:\n\n {incident.resolution}\n\n"
            f"**Total_cost**:\n ${incident.total_cost}\n\n"
            f"**Incident Commander**:\n {incident.commander.individual.name}\n\n"
            f"**Incident Type**:\n {incident.incident_type.name}\n\n"
            f"**Incident Severity**:\n {incident.incident_severity.name}\n\n"
            f"**Incident Participants**:\n\n {incident_participants_str}\n\n"
            f"**Custom Fields**:\n\n {custom_fields_str}\n\n"
        )

        # Create exec summary from openAI
        # Compile the timeline
        timeline = str()
        events = incident.events
        for e in events:
            timeline += (
                f"{e.started_at.replace(second=0, microsecond=0)}        {e.description}\n\n"
            )
        total_chunks = event_service.calculate_total_chunks(input_text=timeline)

        # TTX Metrics
        # Leverage AI to generate findings for the Postmortem/PIR
        story.add_text("# TTX Metrics and Findings")
        prompt = "Here are chat logs during an incident. Based on the timeline, generate the following as seperate sections in markdown format: 1) How long it took to find the root cause? 2) How long it took to resolve the incident? 3) How long it took to communicate the incident to the stakeholders?"
        result = event_service.summarize_text(
            input_text=timeline,
            total_chunks=total_chunks,
            prompt=prompt,
            db_session=db_session,
            project_id=incident.project.id,
        )
        story.add_text(result)

        # Exec Summary
        story.add_text("# Executive R/T NLP Summary")
        prompt = prompt_service.return_most_appropriate_prompt(incident=incident, type=ReportTypes.summary)
        result = event_service.summarize_text(
            input_text=timeline,
            total_chunks=total_chunks,
            prompt=prompt,
            db_session=db_session,
            project_id=incident.project.id,
        )
        story.add_text(result)

        # Timeline and Sequence of Events
        story.add_text("### Timeline and Sequence of Events")
        prompt = "Here are chat logs during an incident. Generate findings for the timeline sequence of events with the most critical events only in markdown format"
        result = event_service.summarize_text(
            input_text=timeline,
            total_chunks=total_chunks,
            prompt=prompt,
            db_session=db_session,
            project_id=incident.project.id,
        )
        story.add_text(result)

        # Add timeline and important events
        story.add_text("# Hypotheses and Critical Timeline Events")
        story.add_view(important_events_view)

        # Add all events.
        story.add_text("## All Events")
        story.add_view(all_events_view)

        # Add graphs. TODO.

        # Leverage AI to generate findings for the Postmortem/PIR
        # Get the user configured postmortem fields
        story.add_text("# Incident Analysis")
        postmortem_fields = (
            incident.incident_type.postmortem_custom_fields
            if incident.incident_type.postmortem_custom_fields
            else []
        )
        for field in postmortem_fields:
            prompt = prompt_config.PROMPT_POSTMORTEM_CUSTOM_FIELDS_OPSGPT_1.format(
                field=field["name"]
            )
            result = event_service.summarize_text(
                input_text=timeline,
                total_chunks=total_chunks,
                prompt=prompt,
                db_session=db_session,
                project_id=incident.project.id,
            )
            story.add_text(f'### {field["name"]}?')
            story.add_text(result)

        # Add action items
        story.add_text("# Action Items")
        for task in incident.tasks:
            if task.type == TaskType.task:
                if task.status == TaskStatus.resolved:
                    story.add_text(f"- [x]  {task.description}")
                else:
                    story.add_text(f"- [ ]  {task.description}")

        story.add_text("# Follow-Ups")
        for task in incident.tasks:
            if task.type == TaskType.follow_up:
                if task.status == TaskStatus.resolved:
                    story.add_text(f"- [x]  {task.description}")
                else:
                    story.add_text(f"- [ ]  {task.description}")

        # Sentiment Breakdown
        story.add_text("## Comms Sentiment Breakdown")
        params = {
            "field": "Sentiment",
            "supported_charts": "hbarchart",
            "chart_title": "Comms Sentiment Breakdown",
            "limit": 20,
        }

        aggr_obj = sketch.store_aggregation(
            name="Comms Sentiment Breakdown",
            description="Created automatically by StoryCurve",
            aggregator_name="field_bucket",
            chart_type="hbarchart",
            aggregator_parameters=params,
        )
        story.add_aggregation(agg_obj=aggr_obj, chart_type="hbarchart")

        # Add Participant Activity
        story.add_text("## Participant Activity")

        # Tags used in the timeline(s)
        story.add_text("## Timeline(s) Tag Breakdown")
        params = {
            "field": "tag",
            "supported_charts": "hbarchart",
            "chart_title": "Timeline(s) Tag Breakdown",
            "limit": 20,
        }

        aggr_obj = sketch.store_aggregation(
            name="Timeline(s) Tag Breakdown",
            description="Created automatically by StoryCurve",
            aggregator_name="field_bucket",
            chart_type="hbarchart",
            aggregator_parameters=params,
        )
        story.add_aggregation(agg_obj=aggr_obj, chart_type="hbarchart")

        # Timeline Breakdown by Source
        story.add_text("## Timeline Breakdown by Source")
        params = {
            "field": "source",
            "supported_charts": "hbarchart",
            "chart_title": "Incident Timeline Events by Source",
            "limit": 20,
        }

        aggr_obj = sketch.store_aggregation(
            name="Incident Timeline Events by Source",
            description="Created automatically by StoryCurve",
            aggregator_name="field_bucket",
            chart_type="hbarchart",
            aggregator_parameters=params,
        )
        story.add_aggregation(agg_obj=aggr_obj, chart_type="hbarchart")

        # Timeline Breakdown by user_id
        story.add_text("## Timeline Breakdown by User")
        params = {
            "field": "user_id",
            "supported_charts": "hbarchart",
            "chart_title": "Incident Timeline Events by User",
            "limit": 20,
        }

        aggr_obj = sketch.store_aggregation(
            name="Incident Timeline Events by User",
            description="Created automatically by StoryCurve",
            aggregator_name="field_bucket",
            chart_type="hbarchart",
            aggregator_parameters=params,
        )
        story.add_aggregation(agg_obj=aggr_obj, chart_type="hbarchart")

        # Tags used in the incident
        story.add_text("## Tags used in the incident")
        for tag in incident.tags:
            story.add_text(f"`{tag.name}`")

        # Add all events.
        story.add_text("## All Events")
        story.add_view(all_events_view)

        # Share sketch with the incident participants.
        share_sketch_with_participants(incident)
        return True, None
    except Exception as e:
        log.exception(e)
        return False, e


def share_sketch_with_participants(incident: Incident):
    """Share sketch with participants"""
    sc_client = config.get_client()
    sketch = sc_client.get_sketch(incident.storycurve_sketch_id)
    participants = []
    for p in incident.participants:
        participants.append(p.individual.email)
    sketch.add_to_acl(participants)


def create_postmortem_conference_event(incident: Incident, db_session: SessionLocal):
    plugin = plugin_service.get_active_instance(
        db_session=db_session,
        plugin_type="conference",
        project_id=incident.project.id,
    )

    if not plugin:
        log.warning("No conference plugin enabled. Skipping creation of conference")
        return

    participants = []
    for p in incident.participants:
        participants.append(p.individual.email)

    # don't want the rest of flow to fail.
    # Create the ext conf obj
    try:
        external_conference = plugin.instance.create(
            name=f"[Postmortem] - {incident.title}",
            description=incident.description,
            title=f"[Postmortem] - {incident.title}",
            participants=participants,
        )
    except Exception as e:
        event_service.log_incident_event(
            db_session=db_session,
            source=plugin.instance.title,
            description="Incident Postmortem Conference Creation Failed",
            incident_id=incident.id,
        )
        log.exception(e)

    event_service.log_incident_event(
        db_session=db_session,
        source=plugin.instance.title,
        description="Incident Postmortem Conference Created",
        incident_id=incident.id,
        details={
            "weblink": external_conference["weblink"],
            "id": external_conference["id"],
        },
        type=EventType.resource_updated,
    )

    external_conference.update(
        {"resource_type": plugin.plugin.slug, "resource_id": external_conference["id"]}
    )

    # we create the internal conference obj
    conference_in = ConferenceCreate(
        resource_id=external_conference["resource_id"],
        resource_type=external_conference["resource_type"],
        weblink=external_conference["weblink"],
        conference_id=external_conference["id"],
        conference_challenge=external_conference["challenge"],
    )

    conference_obj_int = conference_service.create(
        conference_in=conference_in, db_session=db_session
    )

    return conference_obj_int


def create_conversation(incident: Incident, db_session: SessionLocal):
    """Create external communication conversation."""
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )
    conversation = plugin.instance.create(incident.name)
    conversation.update({"resource_type": plugin.plugin.slug, "resource_id": conversation["name"]})

    event_service.log_incident_event(
        db_session=db_session,
        source=plugin.plugin.title,
        description="Incident conversation created",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )

    return conversation


def set_conversation_bookmarks(incident: Incident, db_session: SessionLocal):
    """Sets the conversation bookmarks."""
    if not incident.conversation:
        log.warning("Conversation bookmark not set. No conversation available for this incident.")
        return

    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )
    if not plugin:
        log.warning("Bookmarks not created. No conversation plugin enabled.")
        return

    try:
        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            f"{STORYCURVE_UI_URL}/sketch/{incident.storycurve_sketch_id}/explore?timeline={incident.storycurve_sketch_id}",
            title="Timeline Analysis",
        ) if incident.storycurve_sketch_id else log.warning(
            "StoryCurve bookmark not set. No StoryCurve resource for this incident."
        )

        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            f"{DISPATCH_UI_URL}/{incident.project.organization.slug}/incidents/{incident.name}",
            title="Ticket (Internal)",
        ) if incident.name else log.warning(
            "Incident WEB UI not set. No name available for the incident"
        )

        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            resolve_attr(incident, "incident_document.weblink"),
            title="Incident Document",
        ) if incident.documents else log.warning(
            "Document bookmark not set. No document available for this incident."
        )

        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            resolve_attr(incident, "conference.weblink"),
            title="Video Conference",
        ) if incident.conference else log.warning(
            "Conference bookmark not set. No conference available for this incident."
        )

        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            resolve_attr(incident, "storage.weblink"),
            title="Storage",
        ) if incident.storage else log.warning(
            "Storage bookmark not set. No storage available for this incident."
        )

        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            resolve_attr(incident, "ticket.weblink"),
            title="Ticket (External)",
        ) if incident.ticket else log.warning(
            "Ticket bookmark not set. No ticket available for this incident."
        )

        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            f"{UNALERT_UI_URL}",
            title="OnCall",
        )

        plugin.instance.set_bookmark(
            incident.conversation.channel_id,
            f"{UNSTATUS_UI_URL}",
            title="StatusPage",
        )

        event_service.log_incident_event(
            db_session=db_session,
            source=plugin.plugin.title,
            description="Incident conversation bookmarks set",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )

    except Exception as e:
        event_service.log_incident_event(
            db_session=db_session,
            source="unStruct Core App",
            description=f"Setting the incident conversation bookmarks failed. Reason: {e}",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )
        log.exception(e)


def set_conversation_topic(incident: Incident, db_session: SessionLocal):
    """Sets the conversation topic."""
    if not incident.conversation:
        log.warning("Conversation topic not set. No conversation available for this incident.")
        return

    conversation_topic = (
        f":helmet_with_white_cross: {incident.commander.individual.name}, {incident.commander.team} - "
        f"Type: {incident.incident_type.name} - "
        f"Severity: {incident.incident_severity.name} - "
        f"Priority: {incident.incident_priority.name} - "
        f"Status: {incident.status}"
    )

    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )

    try:
        plugin.instance.set_topic(incident.conversation.channel_id, conversation_topic)
        event_service.log_incident_event(
            db_session=db_session,
            source=plugin.plugin.title,
            description="Incident conversation topic set",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )
    except Exception as e:
        event_service.log_incident_event(
            db_session=db_session,
            source="unStruct Core App",
            description=f"Setting the event conversation topic failed. Reason: {e}",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )
        log.exception(e)


def add_participants_to_conversation(
    participant_emails: List[str], incident: Incident, db_session: SessionLocal
):
    """Adds one or more participants to the conversation."""
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )

    if plugin:
        try:
            plugin.instance.add(incident.conversation.channel_id, participant_emails)
        except Exception as e:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Adding participant(s) to event conversation failed. Reason: {e}",
                incident_id=incident.id,
            )
            log.exception(e)


def add_participant_to_tactical_group(
    user_email: str, incident: Incident, db_session: SessionLocal
):
    """Adds participant to the tactical group."""
    # we get the tactical group
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="participant-group"
    )
    if plugin:
        tactical_group = group_service.get_by_incident_id_and_resource_type(
            db_session=db_session,
            incident_id=incident.id,
            resource_type=f"{plugin.plugin.slug}-tactical-group",
        )
        if tactical_group:
            plugin.instance.add(tactical_group.email, [user_email])


def remove_participant_from_tactical_group(
    user_email: str, incident: Incident, db_session: SessionLocal
):
    """Removes participant from the tactical group."""
    # we get the tactical group
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="participant-group"
    )
    if plugin:
        tactical_group = group_service.get_by_incident_id_and_resource_type(
            db_session=db_session,
            incident_id=incident.id,
            resource_type=f"{plugin.plugin.slug}-tactical-group",
        )
        if tactical_group:
            plugin.instance.remove(tactical_group.email, [user_email])


@background_task
def incident_create_stable_flow(
    *, incident_id: int, organization_slug: str = None, db_session=None
):
    """Creates all resources necessary when an incident is created as 'stable'."""
    incident_create_flow(
        incident_id=incident_id, organization_slug=organization_slug, db_session=db_session
    )
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)
    incident_stable_status_flow(incident=incident, db_session=db_session)


@background_task
def incident_create_closed_flow(
    *, incident_id: int, organization_slug: str = None, db_session=None
):
    """Creates all resources necessary when an incident is created as 'closed'."""
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    # we inactivate all participants
    inactivate_incident_participants(incident, db_session)

    # we set the stable and close times to the reported time
    incident.stable_at = incident.closed_at = incident.reported_at

    update_external_incident_ticket(incident.id, db_session)

    db_session.add(incident)
    db_session.commit()


@background_task
def incident_create_flow(
    *, organization_slug: str, incident_id: int, parent_id: str = None, db_session=None
) -> Incident:
    """Creates all resources required for new incidents."""

    # KJ, first things's first, post to the status page.
    # TODO, based on rules.
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    statuspage_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="statuspage"
    )

    if statuspage_plugin and incident.incident_priority.auto_post_to_status_page:
        # We let AI do the work here. TODO. openai is async andI want it like that. For now, we'll just.
        ai_response_text = None
        prompt = prompt_config.PROMPT_STATUSPAGE_COMMS_OPSGPT_1.format(
            incident.title, incident.description
        )
        ai_plugin = plugin_service.get_preferred_active_instance(
            db_session=db_session,
            project_id=incident.project.id,
            plugin_type="artificial-intelligence",
        )

        if ai_plugin:
            try:
                ai_response_text = ai_plugin.instance.ask(prompt=prompt).strip()
                log.debug(f"AI plugin responded with: {ai_response_text}")
            except Exception as e:
                log.error(f"AI plugin failed to respond. Reason: {e}")
        else:
            log.warning(
                "No AI plugin enabled. Unable to generate status page description. Will use incident description."
            )

        if ai_response_text is not None:
            description = ai_response_text
            log.debug(f"AI plugin responded with: {description}")
        else:
            # We use the incident description as the status page description. TODO. Use generic?
            description = incident.description

        try:
            statuspage_id = statuspage_plugin.instance.post_incident(
                title=incident.title, description=description
            )
            if statuspage_id is not None:
                incident.statuspage_id = statuspage_id
                event_service.log_incident_event(
                    db_session=db_session,
                    source=statuspage_plugin.plugin.title,
                    description="Incident posted to the status page",
                    incident_id=incident.id,
                    type=EventType.resource_updated,
                )
            else:
                event_service.log_incident_event(
                    db_session=db_session,
                    source=statuspage_plugin.plugin.title,
                    description="Incident not posted to the status page. Check the plugin configuration.",
                    incident_id=incident.id,
                    type=EventType.resource_updated,
                )
        except Exception as e:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Posting to the status page failed. Reason: {e}",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
            log.exception(e)
    else:
        log.warning(
            "Incident not posted to the status page. No status page plugin enabled or auto_post_to_status_page is false"
        )

    # Second, extract tags and add them to the incident.
    # KJ. TODO.

    # Get additional participants from the incident's project search filters and rules
    individual_participants, team_participants = get_incident_participants(incident, db_session)

    tactical_group = notifications_group = None
    group_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="participant-group"
    )
    if group_plugin:
        try:
            tactical_group_external, notifications_group_external = create_participant_groups(
                incident, individual_participants, team_participants, db_session
            )

            if tactical_group_external and notifications_group_external:
                tactical_group_in = GroupCreate(
                    name=tactical_group_external["name"],
                    email=tactical_group_external["email"],
                    resource_type=tactical_group_external["resource_type"],
                    resource_id=tactical_group_external["resource_id"],
                    weblink=tactical_group_external["weblink"],
                )
                tactical_group = group_service.create(
                    db_session=db_session, group_in=tactical_group_in
                )
                incident.groups.append(tactical_group)
                incident.tactical_group_id = tactical_group.id

                notifications_group_in = GroupCreate(
                    name=notifications_group_external["name"],
                    email=notifications_group_external["email"],
                    resource_type=notifications_group_external["resource_type"],
                    resource_id=notifications_group_external["resource_id"],
                    weblink=notifications_group_external["weblink"],
                )
                notifications_group = group_service.create(
                    db_session=db_session, group_in=notifications_group_in
                )
                incident.groups.append(notifications_group)
                incident.notifications_group_id = notifications_group.id

                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Tactical and notifications groups added to incident",
                    incident_id=incident.id,
                    type=EventType.resource_updated,
                )
        except Exception as e:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Creation of tactical and notifications groups failed. Reason: {e}",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
            log.exception(e)

    storage_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="storage"
    )
    if storage_plugin:
        # we create the storage resource
        try:
            if group_plugin:
                group_emails = []
                if tactical_group and notifications_group:
                    group_emails = [tactical_group.email, notifications_group.email]

                storage = create_incident_storage(incident, group_emails, db_session)
            else:
                participant_emails = [x.email for x, _ in individual_participants]

                # we don't have a group so add participants directly
                storage = create_incident_storage(incident, participant_emails, db_session)

            storage_in = StorageCreate(
                resource_id=storage["resource_id"],
                resource_type=storage["resource_type"],
                weblink=storage["weblink"],
            )

            incident.storage = storage_service.create(
                db_session=db_session,
                storage_in=storage_in,
            )

            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Storage added to incident",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
        except Exception as e:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Creation of event storage failed. Reason: {e}",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
            log.exception(e)

        # we create collaboration documents, don't fail the whole flow if this fails
        try:
            incident_documents = create_incident_documents(incident, db_session)

            for d in incident_documents:
                document_in = DocumentCreate(
                    name=d["name"],
                    description=d["description"],
                    resource_id=d["resource_id"],
                    project={"name": incident.project.name},
                    resource_type=d["resource_type"],
                    weblink=d["weblink"],
                )
                document = document_service.create(db_session=db_session, document_in=document_in)
                incident.documents.append(document)

                if document.resource_type == DocumentResourceTypes.incident:
                    incident.incident_document_id = document.id

            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Collaboration documents added to incident",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
        except Exception as e:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Creation of collaboration documents failed. Reason: {e}",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
            log.exception(e)

    conference_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conference"
    )
    if conference_plugin:
        try:
            participant_emails = [x.email for x, _ in individual_participants]

            if group_plugin and tactical_group:
                # we use the tactical group email if the group plugin is enabled
                participant_emails = [tactical_group.email]

            conference = create_conference(incident, participant_emails, db_session)

            conference_in = ConferenceCreate(
                resource_id=conference["resource_id"],
                resource_type=conference["resource_type"],
                weblink=conference["weblink"],
                conference_id=conference["id"],
                conference_challenge=conference["challenge"],
            )
            incident.conference = conference_service.create(
                db_session=db_session, conference_in=conference_in
            )

            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Conference added to incident",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
        except Exception as e:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Creation of event conference failed. Reason: {e}",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
            log.exception(e)

    # we create the conversation for real-time communications

    conversation_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )
    if conversation_plugin:
        try:
            conversation = create_conversation(incident, db_session)

            conversation_in = ConversationCreate(
                resource_id=conversation["resource_id"],
                resource_type=conversation["resource_type"],
                weblink=conversation["weblink"],
                description=INCIDENT_CONVERSATION_DESCRIPTION,
                channel_id=conversation["id"],
            )
            incident.conversation = conversation_service.create(
                db_session=db_session, conversation_in=conversation_in
            )

            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Conversation added to incident",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )

            # we set the conversation topic
            set_conversation_topic(incident, db_session)
            # we set the conversation bookmarks
            set_conversation_bookmarks(incident, db_session)
        except Exception as e:
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description=f"Creation of incident conversation failed. Reason: {e}",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
            log.exception(e)

    # we update the incident ticket
    update_external_incident_ticket(incident.id, db_session)

    # we update the investigation document
    document_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="document"
    )
    if document_plugin:
        if incident.incident_document:
            try:
                document_plugin.instance.update(
                    incident.incident_document.resource_id,
                    commander_fullname=incident.commander.individual.name,
                    conference_challenge=resolve_attr(incident, "conference.challenge"),
                    conference_weblink=resolve_attr(incident, "conference.weblink"),
                    conversation_weblink=resolve_attr(incident, "conversation.weblink"),
                    description=incident.description,
                    document_weblink=resolve_attr(incident, "incident_document.weblink"),
                    name=incident.name,
                    priority=incident.incident_priority.name,
                    severity=incident.incident_severity.name,
                    status=incident.status,
                    storage_weblink=resolve_attr(incident, "storage.weblink"),
                    ticket_weblink=resolve_attr(incident, "ticket.weblink"),
                    title=incident.title,
                    type=incident.incident_type.name,
                )
            except Exception as e:
                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description=f"Event documents rendering failed. Reason: {e}",
                    incident_id=incident.id,
                    type=EventType.resource_updated,
                )
                log.exception(e)

    # we defer this setup for all resolved incident roles until after resources have been created
    roles = ["reporter", "commander", "liaison", "scribe"]

    user_emails = [
        resolve_attr(incident, f"{role}.individual.email")
        for role in roles
        if resolve_attr(incident, role)
    ]
    user_emails = list(dict.fromkeys(user_emails))

    for user_email in user_emails:
        # we add the participant to the tactical group
        add_participant_to_tactical_group(user_email, incident, db_session)

        # we add the participant to the conversation
        add_participants_to_conversation([user_email], incident, db_session)

        # we announce the participant in the conversation
        send_incident_participant_announcement_message(user_email, incident, db_session)

        # we send the welcome messages to the participant
        send_incident_welcome_participant_messages(user_email, incident, db_session)

        # we send a suggested reading message to the participant
        suggested_document_items = get_suggested_document_items(incident, db_session)
        send_incident_suggested_reading_messages(
            incident, suggested_document_items, user_email, db_session
        )

    # wait until all resources are created before adding suggested participants
    for individual, service_id in individual_participants:
        incident_add_or_reactivate_participant_flow(
            individual.email,
            incident.id,
            participant_role=ParticipantRoleType.observer,
            service_id=service_id,
            db_session=db_session,
        )

    event_service.log_incident_event(
        db_session=db_session,
        source="unStruct Core App",
        description="Incident participants added to incident",
        incident_id=incident.id,
        type=EventType.resource_updated,
    )

    send_incident_created_notifications(incident, db_session)

    event_service.log_incident_event(
        db_session=db_session,
        source="unStruct Core App",
        description="Incident notifications sent",
        incident_id=incident.id,
        type=EventType.notification,
    )

    # we page the incident commander based on incident priority
    if incident.incident_priority.page_commander:
        if incident.commander.service:
            service_id = incident.commander.service.external_id
            oncall_plugin = plugin_service.get_active_instance(
                db_session=db_session, project_id=incident.project.id, plugin_type="oncall"
            )
            if oncall_plugin:
                try:
                    oncall_plugin.instance.page(
                        service_id=service_id,
                        incident_name=incident.name,
                        incident_title=incident.title,
                        incident_description=incident.description,
                    )
                except Exception as e:
                    log.exception(e)
                    event_service.log_incident_event(
                        db_session=db_session,
                        source="unStruct Core App",
                        description=f"Failed to page incident commander. Reason: {e}",
                        incident_id=incident.id,
                        type=EventType.notification,
                    )
                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Incident Commander Paged",
                    incident_id=incident.id,
                    type=EventType.notification,
                )
            else:
                log.warning("Incident commander not paged. No plugin of type oncall enabled.")
                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Incident commander not paged. No plugin of type oncall enabled",
                    incident_id=incident.id,
                    type=EventType.notification,
                )
        else:
            log.warning(
                "Incident commander not paged. No relationship between commander and an oncall service."
            )
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Incident commander not paged. No relationship between commander and an oncall service",
                incident_id=incident.id,
                type=EventType.notification,
            )

    # we page the liaison based on incident priority
    if incident.incident_priority.page_liaison:
        if incident.liaison.service:
            service_id = incident.liaison.service.external_id
            oncall_plugin = plugin_service.get_active_instance(
                db_session=db_session, project_id=incident.project.id, plugin_type="oncall"
            )
            if oncall_plugin:
                try:
                    oncall_plugin.instance.page(
                        service_id=service_id,
                        incident_name=incident.name,
                        incident_title=incident.title,
                        incident_description=incident.description,
                    )
                except Exception as e:
                    log.exception(e)
                    event_service.log_incident_event(
                        db_session=db_session,
                        source="unStruct Core App",
                        description=f"Failed to page liaison. Reason: {e}",
                        incident_id=incident.id,
                        type=EventType.notification,
                    )
                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Liaison Paged",
                    incident_id=incident.id,
                    type=EventType.notification,
                )
            else:
                log.warning("Liaison not paged. No plugin of type oncall enabled.")
                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Liaison not paged. No plugin of type oncall enabled",
                    incident_id=incident.id,
                    type=EventType.notification,
                )
        else:
            log.warning("Liaison not paged. No relationship between liaison and an oncall service.")
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Liaison not paged. No relationship between liaison and an oncall service",
                incident_id=incident.id,
                type=EventType.notification,
            )

    # we page the scribe based on incident priority
    if incident.incident_priority.page_scribe:
        if incident.scribe.service:
            service_id = incident.scribe.service.external_id
            oncall_plugin = plugin_service.get_active_instance(
                db_session=db_session, project_id=incident.project.id, plugin_type="oncall"
            )
            if oncall_plugin:
                try:
                    oncall_plugin.instance.page(
                        service_id=service_id,
                        incident_name=incident.name,
                        incident_title=incident.title,
                        incident_description=incident.description,
                    )
                except Exception as e:
                    log.exception(e)
                    event_service.log_incident_event(
                        db_session=db_session,
                        source="unStruct Core App",
                        description=f"Failed to page scribe. Reason: {e}",
                        incident_id=incident.id,
                        type=EventType.notification,
                    )
                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Scribe Paged",
                    incident_id=incident.id,
                    type=EventType.notification,
                )
            else:
                log.warning("Scribe not paged. No plugin of type oncall enabled.")
                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Scribe not paged. No plugin of type oncall enabled",
                    incident_id=incident.id,
                    type=EventType.notification,
                )
        else:
            log.warning("Scribe not paged. No relationship between scribe and an oncall service.")
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Scribe not paged. No relationship between scribe and an oncall service",
                incident_id=incident.id,
                type=EventType.notification,
            )

    # we send a message to the incident commander with tips on how to manage the incident
    send_incident_management_help_tips_message(incident, db_session)

    # We share the sketch with all participants
    share_sketch_with_participants(incident)

    # We extract terms from the incident title and description
    extracted_tags = extract_tags(incident.title) + extract_tags(incident.description)
    incident_service.add_tags_to_incident(
        incident=incident,
        tags_list=extracted_tags,
        db_session=db_session,
        tag_description="Auto-generated from parsing Incident Metadata/Reports",
        tag_type_name="NLP",
    )

    db_session.add(incident)
    db_session.commit()

    return incident


def incident_active_status_flow(incident: Incident, db_session=None):
    """Runs the incident active flow."""
    # we un-archive the conversation
    convo_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )
    if convo_plugin:
        convo_plugin.instance.unarchive(incident.conversation.channel_id)


def incident_stable_status_flow(incident: Incident, db_session=None):
    """Runs the incident stable flow."""
    # we set the stable time
    incident.stable_at = datetime.utcnow()
    db_session.add(incident)
    db_session.commit()

    # We update the statuspage
    statuspage_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="statuspage"
    )

    # only update if the incident is linked to a statuspage
    if (
        statuspage_plugin
        and incident.statuspage_id
        and incident.incident_priority.auto_post_to_status_page
    ):
        try:
            statuspage_plugin.instance.resolve_incident(
                id=incident.statuspage_id,
                description="This Incident has been marked as stable in unStruct",
            )
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Incident status updated in statuspage",
                incident_id=incident.id,
                type=EventType.notification,
            )
        except Exception as e:
            log.warning(f"Failed to update incident in the status page {e}")

    if incident.incident_document:
        # we update the incident document
        update_document(incident.incident_document.resource_id, incident, db_session)

    if incident.incident_review_document:
        log.debug(
            "The post-incident review document has already been created. Skipping creation..."
        )
        return

    # we create the post-incident review document and also the storycurve sketch
    create_post_incident_review_document(incident, db_session)

    if incident.incident_review_document:
        # we send a notification about the incident review document to the conversation
        send_incident_review_document_notification(
            incident.conversation.channel_id,
            incident.incident_review_document.weblink,
            incident,
            db_session,
        )

    # We create a calendar invite for the participants to conduct a debrief/postmortem
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conference"
    )
    if plugin and incident.incident_priority.auto_schedule_postmortem_meeting:
        conference = create_postmortem_conference_event(incident, db_session)

        # we send a notification about the postmortem meeting to the conversation
        if conference.weblink:
            # Notify the channel that the posmortem meeting is ready, so they can edit
            send_postmortem_conference_notification(
                incident.conversation.channel_id,
                conference.weblink,
                incident,
                db_session,
            )

            # Add the conference to the postmortem object
            postmortem = postmortem_service.get_by_incident_id(
                db_session=db_session, incident_id=incident.id
            )

            # CHeck if we have a postmortem object
            if postmortem:
                postmortem.conference = conference
                db_session.add(postmortem)
                db_session.commit()
            else:
                postmortem = None
                log.debug("Postmortem object not found. Skipping conference addition.")
        else:
            log.debug("Postmortem conference creation failed.")
    else:
        conference = None
        log.debug(
            "Postmortem conference not created. No conference plugin enabled or config not enabled."
        )

    # we bookmark the incident review document
    plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )
    if not plugin:
        log.warning("Incident review document not bookmarked. No conversation plugin enabled.")
        return

    # we bookmark the incident review document and the postmortem conference
    try:
        # Check if we have a review document
        if incident.incident_review_document:
            plugin.instance.set_bookmark(
                incident.conversation.channel_id,
                resolve_attr(incident, "incident_review_document.weblink"),
                title="Review Document",
            )
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Incident review document bookmarked",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )

        # Check if we have conference object
        if conference:
            plugin.instance.set_bookmark(
                incident.conversation.channel_id, conference.weblink, title="Postmortem Conference"
            )
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Postmortem conference bookmarked",
                incident_id=incident.id,
                type=EventType.resource_updated,
            )
    except Exception as e:
        event_service.log_incident_event(
            db_session=db_session,
            source="unStruct Core App",
            description=f"Bookmarking the incident review document failed. Reason: {e}",
            incident_id=incident.id,
            type=EventType.resource_updated,
        )
        log.exception(e)


def incident_closed_status_flow(incident: Incident, db_session=None):
    """Runs the incident closed flow."""
    # we inactivate all participants
    inactivate_incident_participants(incident, db_session)

    # we set the closed time
    incident.closed_at = datetime.utcnow()
    db_session.add(incident)
    db_session.commit()

    # We update the statuspage
    statuspage_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="statuspage"
    )

    # only update if the incident is linked to a statuspage
    if (
        statuspage_plugin
        and incident.statuspage_id
        and incident.incident_priority.auto_post_to_status_page
    ):
        try:
            statuspage_plugin.instance.close_incident(
                id=incident.statuspage_id,
                description="This incident has been marked as closed in unStruct",
            )
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Incident status updated in statuspage",
                incident_id=incident.id,
                type=EventType.notification,
            )
        except Exception as e:
            log.warning(f"Failed to update incident in the status page {e}")

    # we archive the conversation
    # TODO. We need to make it configurable. KJ.
    convo_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="conversation"
    )
    if convo_plugin:
        convo_plugin.instance.archive(incident.conversation.channel_id)

    # storage for incidents with restricted visibility is never opened
    if incident.visibility == Visibility.open:
        # add organization wide permission
        storage_plugin = plugin_service.get_active_instance(
            db_session=db_session, project_id=incident.project.id, plugin_type="storage"
        )
        if storage_plugin:
            if storage_plugin.configuration.open_on_close:
                # typically only broad access to the incident document itself is required.
                storage_plugin.instance.open(incident.incident_document.resource_id)

                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Incident document opened to anyone in the domain",
                    incident_id=incident.id,
                    type=EventType.imported_message,
                )

            if storage_plugin.configuration.read_only:
                # unfortunately this can't be applied at the folder level
                # so we just mark the incident doc as available.
                storage_plugin.instance.mark_readonly(incident.incident_document.resource_id)

                event_service.log_incident_event(
                    db_session=db_session,
                    source="unStruct Core App",
                    description="Incident document marked as readonly",
                    incident_id=incident.id,
                    type=EventType.imported_message,
                )

    # we send a direct message to the incident commander asking to review
    # the incident's information and to tag the incident if appropiate
    send_incident_closed_information_review_reminder(incident, db_session)

    # we send a direct message to all participants asking them
    # to rate and provide feedback about the incident
    send_incident_rating_feedback_message(incident, db_session)

    # TODO. For now, just run the postmortem flow
    incident_postmortem_completed_status_flow(incident, db_session)


def incident_postmortem_completed_status_flow(incident: Incident, db_session=None):
    """Runs the incident postmortem completed flow."""

    # We update the statuspage
    statuspage_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="statuspage"
    )

    # only update if the incident is linked to a statuspage
    if (
        statuspage_plugin
        and incident.statuspage_id
        and incident.incident_priority.auto_post_to_status_page
    ):
        try:
            statuspage_plugin.instance.postmortem_completed(
                id=incident.statuspage_id,
                description=f"This Postmortem has been completed for this incident in unStruct. Here is the link: {STORYCURVE_UI_URL}/sketch/{incident.storycurve_sketch_id}/explore?timeline={incident.storycurve_sketch_id}.",
            )
            event_service.log_incident_event(
                db_session=db_session,
                source="unStruct Core App",
                description="Incident postmortem comlpletion status updated in statuspage",
                incident_id=incident.id,
                type=EventType.notification,
            )
        except Exception as e:
            log.warning(f"Failed to update incident postmortem status in the status page {e}")

    # Send the postmortem completed message to the participants
    incident_service.send_incident_postmortem_completed_message(
        incident=incident, db_session=db_session
    )


def conversation_topic_dispatcher(
    user_email: str,
    incident: Incident,
    previous_incident: dict,
    db_session: SessionLocal,
):
    """Determines if the conversation topic needs to be updated."""
    # we load the individual
    individual = individual_service.get_by_email_and_project(
        db_session=db_session, email=user_email, project_id=incident.project.id
    )

    conversation_topic_change = False
    if previous_incident.title != incident.title:
        event_service.log_incident_event(
            db_session=db_session,
            source="Incident Participant",
            description=f'{individual.name} changed the incident title to "{incident.title}"',
            incident_id=incident.id,
            individual_id=individual.id,
            type=EventType.field_updated,
        )

    if previous_incident.description != incident.description:
        event_service.log_incident_event(
            db_session=db_session,
            source="Incident Participant",
            description=f"{individual.name} changed the incident description",
            details={"description": incident.description},
            incident_id=incident.id,
            individual_id=individual.id,
            type=EventType.field_updated,
        )

    if previous_incident.incident_type.name != incident.incident_type.name:
        conversation_topic_change = True

        event_service.log_incident_event(
            db_session=db_session,
            source="Incident Participant",
            description=f"{individual.name} changed the incident type to {incident.incident_type.name}",
            incident_id=incident.id,
            individual_id=individual.id,
            type=EventType.field_updated,
        )

    if previous_incident.incident_severity.name != incident.incident_severity.name:
        conversation_topic_change = True

        event_service.log_incident_event(
            db_session=db_session,
            source="Incident Participant",
            description=f"{individual.name} changed the incident severity to {incident.incident_severity.name}",
            incident_id=incident.id,
            individual_id=individual.id,
            type=EventType.field_updated,
        )

    if previous_incident.incident_priority.name != incident.incident_priority.name:
        conversation_topic_change = True

        event_service.log_incident_event(
            db_session=db_session,
            source="Incident Participant",
            description=f"{individual.name} changed the incident priority to {incident.incident_priority.name}",
            incident_id=incident.id,
            individual_id=individual.id,
            type=EventType.field_updated,
        )

    if previous_incident.status != incident.status:
        conversation_topic_change = True

        event_service.log_incident_event(
            db_session=db_session,
            source="Incident Participant",
            description=f"{individual.name} marked the incident as {incident.status.lower()}",
            incident_id=incident.id,
            individual_id=individual.id,
            type=EventType.field_updated,
        )

    if conversation_topic_change:
        if incident.status != IncidentStatus.closed:
            set_conversation_topic(incident, db_session)


def status_flow_dispatcher(
    incident: Incident,
    current_status: IncidentStatus,
    previous_status: IncidentStatus,
    db_session=SessionLocal,
):
    """Runs the correct flows depending on the incident's current and previous status."""
    # we have a currently active incident
    if current_status == IncidentStatus.active:
        if previous_status == IncidentStatus.closed:
            # re-activate incident
            incident_active_status_flow(incident=incident, db_session=db_session)
            reactivate_incident_participants(incident=incident, db_session=db_session)
            send_incident_report_reminder(incident, ReportTypes.tactical_report, db_session)
        elif previous_status == IncidentStatus.stable:
            send_incident_report_reminder(incident, ReportTypes.tactical_report, db_session)

    # we currently have a stable incident
    elif current_status == IncidentStatus.stable:
        if previous_status == IncidentStatus.active:
            incident_stable_status_flow(incident=incident, db_session=db_session)
            send_incident_report_reminder(incident, ReportTypes.tactical_report, db_session)
        elif previous_status == IncidentStatus.closed:
            incident_active_status_flow(incident=incident, db_session=db_session)
            incident_stable_status_flow(incident=incident, db_session=db_session)
            reactivate_incident_participants(incident=incident, db_session=db_session)
            send_incident_report_reminder(incident, ReportTypes.tactical_report, db_session)

    # we currently have a closed incident
    elif current_status == IncidentStatus.closed:
        if previous_status == IncidentStatus.active:
            incident_stable_status_flow(incident=incident, db_session=db_session)
            incident_closed_status_flow(incident=incident, db_session=db_session)
        elif previous_status == IncidentStatus.stable:
            incident_closed_status_flow(incident=incident, db_session=db_session)

    if previous_status != current_status:
        event_service.log_incident_event(
            db_session=db_session,
            source="unStruct Core App",
            description=f"The incident status has been changed from {previous_status.lower()} to {current_status.lower()}",  # noqa
            incident_id=incident.id,
            type=EventType.field_updated,
        )


@background_task
def incident_update_flow(
    user_email: str,
    commander_email: str,
    reporter_email: str,
    assignee_email: str,
    incident_id: int,
    previous_incident: IncidentRead,
    organization_slug: str = None,
    db_session=None,
):
    """Runs the incident update flow."""
    # we load the incident
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    # we update the commander if needed
    incident_assign_role_flow(
        incident_id=incident_id,
        assigner_email=user_email,
        assignee_email=commander_email,
        assignee_role=ParticipantRoleType.incident_commander,
        db_session=db_session,
    )

    # we update the reporter if needed
    incident_assign_role_flow(
        incident_id=incident_id,
        assigner_email=user_email,
        assignee_email=reporter_email,
        assignee_role=ParticipantRoleType.reporter,
        db_session=db_session,
    )

    # We update the assignee if needed
    incident_assign_role_flow(
        incident_id=incident_id,
        assigner_email=user_email,
        assignee_email=assignee_email,
        assignee_role=ParticipantRoleType.assignee,
        db_session=db_session,
    )

    # we run the active, stable or closed flows based on incident status change
    status_flow_dispatcher(
        incident, incident.status, previous_incident.status, db_session=db_session
    )

    # we update the conversation topic
    conversation_topic_dispatcher(user_email, incident, previous_incident, db_session=db_session)

    # we update the external ticket
    update_external_incident_ticket(incident_id, db_session)

    if incident.status == IncidentStatus.active:
        # we re-resolve and add individuals to the incident
        individual_participants, team_participants = get_incident_participants(incident, db_session)

        for individual, service_id in individual_participants:
            incident_add_or_reactivate_participant_flow(
                individual.email,
                incident.id,
                participant_role=ParticipantRoleType.observer,
                service_id=service_id,
                db_session=db_session,
            )

        # we add the team distributions lists to the notifications group
        # we only have to do this for teams as new members
        # will be added to the tactical group on incident join
        group_plugin = plugin_service.get_active_instance(
            db_session=db_session, project_id=incident.project.id, plugin_type="participant-group"
        )
        if group_plugin:
            team_participant_emails = [x.email for x in team_participants]
            group_plugin.instance.add(incident.notifications_group.email, team_participant_emails)

    # we send the incident update notifications
    send_incident_update_notifications(incident, previous_incident, db_session)

    # we share the sketch with all partcipants
    share_sketch_with_participants(incident)


def incident_delete_flow(incident: Incident, db_session: Session):
    """Deletes all external incident resources."""
    # we delete the external ticket
    if incident.ticket:
        ticket_flows.delete_ticket(
            ticket=incident.ticket, project_id=incident.project.id, db_session=db_session
        )

    # we delete the external groups
    if incident.groups:
        for group in incident.groups:
            group_flows.delete_group(
                group=group, project_id=incident.project.id, db_session=db_session
            )

    # we delete the external storage
    if incident.storage:
        storage_flows.delete_storage(
            storage=incident.storage, project_id=incident.project.id, db_session=db_session
        )

    # we delete the conversation
    if incident.conversation:
        conversation_flows.delete_conversation(
            conversation=incident.conversation,
            project_id=incident.project.id,
            db_session=db_session,
        )


def incident_assign_role_flow(
    incident_id: int,
    assigner_email: str,
    assignee_email: str,
    assignee_role: str,
    db_session: SessionLocal,
):
    """Runs the incident participant role assignment flow."""
    # we load the incident instance
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    # we add the assignee to the incident if they're not a participant
    incident_add_or_reactivate_participant_flow(assignee_email, incident.id, db_session=db_session)

    # we run the participant assign role flow
    result = participant_role_flows.assign_role_flow(
        incident, assignee_email, assignee_role, db_session
    )

    if result == "assignee_has_role":
        # NOTE: This is disabled until we can determine the source of the caller
        # we let the assigner know that the assignee already has this role
        # send_incident_participant_has_role_ephemeral_message(
        # 	assigner_email, assignee_contact_info, assignee_role, incident
        # )
        return

    if result == "role_not_assigned":
        # NOTE: This is disabled until we can determine the source of the caller
        # we let the assigner know that we were not able to assign the role
        # send_incident_participant_role_not_assigned_ephemeral_message(
        # 	assigner_email, assignee_contact_info, assignee_role, incident
        # )
        return

    if incident.status != IncidentStatus.closed:
        if assignee_role != ParticipantRoleType.participant:
            # we resolve the assigner and assignee contact information
            contact_plugin = plugin_service.get_active_instance(
                db_session=db_session, project_id=incident.project.id, plugin_type="contact"
            )

            if contact_plugin:
                assigner_contact_info = contact_plugin.instance.get(
                    assigner_email, db_session=db_session, project_id=incident.project.id
                )
                assignee_contact_info = contact_plugin.instance.get(
                    assignee_email, db_session=db_session, project_id=incident.project.id
                )
            else:
                assigner_contact_info = {
                    "email": assigner_email,
                    "fullname": "Unknown",
                    "weblink": "",
                }
                assignee_contact_info = {
                    "email": assignee_email,
                    "fullname": "Unknown",
                    "weblink": "",
                }

            # we send a notification to the incident conversation
            send_incident_new_role_assigned_notification(
                assigner_contact_info, assignee_contact_info, assignee_role, incident, db_session
            )

        if assignee_role == ParticipantRoleType.incident_commander:
            # we update the conversation topic
            set_conversation_topic(incident, db_session)

            # we send a message to the incident commander with tips on how to manage the incident
            send_incident_management_help_tips_message(incident, db_session)


@background_task
def incident_engage_oncall_flow(
    user_email: str,
    incident_id: int,
    oncall_service_external_id: str,
    page=None,
    organization_slug: str = None,
    db_session=None,
):
    """Runs the incident engage oncall flow."""
    # we load the incident instance
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    # we resolve the oncall service
    oncall_service = service_service.get_by_external_id_and_project_id(
        db_session=db_session,
        external_id=oncall_service_external_id,
        project_id=incident.project.id,
    )

    # we get the active oncall plugin
    oncall_plugin = plugin_service.get_active_instance(
        db_session=db_session, project_id=incident.project.id, plugin_type="oncall"
    )

    if oncall_plugin:
        if oncall_plugin.plugin.slug != oncall_service.type:
            log.warning(
                f"Unable to engage the oncall. Oncall plugin enabled not of type {oncall_plugin.plugin.slug}."  # noqa
            )
            return None, None
    else:
        log.warning("Unable to engage the oncall. No oncall plugins enabled.")
        return None, None

    oncall_person = oncall_plugin.instance.get(service_id=oncall_service_external_id)
    oncall_email = oncall_person.get("email")

    # we attempt to add the oncall to the incident
    oncall_participant_added = incident_add_or_reactivate_participant_flow(
        oncall_email, incident.id, service_id=oncall_service.id, db_session=db_session
    )

    if not oncall_participant_added:
        # we already have the oncall for the service in the incident
        return None, oncall_service

    individual = individual_service.get_by_email_and_project(
        db_session=db_session, email=user_email, project_id=incident.project.id
    )

    event_service.log_incident_event(
        db_session=db_session,
        source=oncall_plugin.plugin.title,
        description=f"{individual.name} engages oncall service {oncall_service.name}",
        incident_id=incident.id,
        type=EventType.notification,
    )

    if page == "Yes":
        # we page the oncall
        oncall_plugin.instance.page(
            service_id=oncall_service_external_id,
            incident_name=incident.name,
            incident_title=incident.title,
            incident_description=incident.description,
        )

        event_service.log_incident_event(
            db_session=db_session,
            source=oncall_plugin.plugin.title,
            description=f"{oncall_service.name} on-call paged",
            incident_id=incident.id,
            type=EventType.notification,
        )

    return oncall_participant_added.individual, oncall_service


@background_task
def incident_add_participant_to_tactical_group_flow(
    user_email: str,
    incident_id: Incident,
    organization_slug: str,
    db_session: SessionLocal,
):
    """Adds participant to the tactical group."""
    # we get the tactical group
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    add_participant_to_tactical_group(
        db_session=db_session, incident=incident, user_email=user_email
    )


@background_task
def incident_add_or_reactivate_participant_flow(
    user_email: str,
    incident_id: int,
    participant_role: ParticipantRoleType = ParticipantRoleType.participant,
    service_id: int = 0,
    event: dict = None,
    organization_slug: str = None,
    db_session=None,
) -> Participant:
    """Runs the incident add or reactivate participant flow."""
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    if service_id:
        # we need to ensure that we don't add another member of a service if one
        # already exists (e.g. overlapping oncalls, we assume they will hand-off if necessary)
        participant = participant_service.get_by_incident_id_and_service_id(
            incident_id=incident_id, service_id=service_id, db_session=db_session
        )

        if participant:
            log.debug("Skipping resolved participant. Oncall service member already engaged.")
            return

    participant = participant_service.get_by_incident_id_and_email(
        db_session=db_session, incident_id=incident.id, email=user_email
    )

    if participant:
        if participant.active_roles:
            return participant

        if incident.status != IncidentStatus.closed:
            # we reactivate the participant
            participant_flows.reactivate_participant(
                user_email, incident, db_session, service_id=service_id
            )
    else:
        # we add the participant to the incident
        participant = participant_flows.add_participant(
            user_email, incident, db_session, service_id=service_id, role=participant_role
        )

    # we add the participant to the tactical group
    add_participant_to_tactical_group(user_email, incident, db_session)

    if incident.status != IncidentStatus.closed:
        # we add the participant to the conversation
        add_participants_to_conversation([user_email], incident, db_session)

        # we announce the participant in the conversation
        send_incident_participant_announcement_message(user_email, incident, db_session)

        # we send the welcome messages to the participant
        send_incident_welcome_participant_messages(user_email, incident, db_session)

        # we send a suggested reading message to the participant
        suggested_document_items = get_suggested_document_items(incident, db_session)
        send_incident_suggested_reading_messages(
            incident, suggested_document_items, user_email, db_session
        )

        # We share the sketch with all participants
        share_sketch_with_participants(incident)

    return participant


@background_task
def incident_remove_participant_flow(
    user_email: str,
    incident_id: int,
    event: dict = None,
    organization_slug: str = None,
    db_session=None,
):
    """Runs the remove participant flow."""
    incident = incident_service.get(db_session=db_session, incident_id=incident_id)

    participant = participant_service.get_by_incident_id_and_email(
        db_session=db_session, incident_id=incident.id, email=user_email
    )

    # Fetch all tasks for the incident of type `task`
    tasks = task_service.get_all_by_incident_id_and_type(
        db_session=db_session, incident_id=incident.id, type=TaskType.task
    )
    for task in tasks:
        if task.status == TaskStatus.open:
            if task.owner == participant:
                # we add the participant back to the conversation
                add_participants_to_conversation([user_email], incident, db_session)

                # we ask the participant to resolve or re-assign
                # their tasks before leaving the incident
                send_incident_open_tasks_ephemeral_message(user_email, incident, db_session)

                return

    if user_email == incident.commander.individual.email:
        # we add the incident commander back to the conversation
        add_participants_to_conversation([user_email], incident, db_session)

        # we send a notification to the channel
        send_incident_commander_readded_notification(incident, db_session)

        return

    # we remove the participant from the incident
    participant_flows.remove_participant(user_email, incident, db_session)

    # we remove the participant to the tactical group
    remove_participant_from_tactical_group(user_email, incident, db_session)
