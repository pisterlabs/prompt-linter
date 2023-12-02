from icalendar import Calendar
from langchain.document_loaders.directory import DirectoryLoader
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from typing import Optional


class Event(BaseModel):
    summary: Optional[str] = Field(default=None, allow_none=True)
    description: Optional[str] = Field(default=None, allow_none=True)
    start_time: datetime
    end_time: Optional[datetime] = Field(default=None, allow_none=True)
    page_content: Optional[str] = Field(default=None, allow_none=True)
    metadata: Optional[dict] = Field(default=None, allow_none=True)

    def __init__(self, summary: Optional[str] = None, 
                    description: Optional[str] = None,
                    start_time: datetime = None, 
                    end_time: Optional[datetime] = None,
                    metadata: Optional[dict] = None,
                    ):
        super().__init__(summary=summary, description=description, start_time=start_time, end_time=end_time, metadata=metadata)
        self.page_content = f"{summary or ''} {description or ''}"

class ICalLoader(DirectoryLoader):
    def __init__(self, folder):
        super().__init__(folder, glob='*.ics')
        self.logger = logging.getLogger(__name__)  # Initialize logger


    def _is_visible(self, p):
        parts = p.parts
        for _p in parts:
            if _p.startswith("."):
                return False
        return True

    def load_file(self, item, path, docs, pbar=None):
        if item.is_file():
            if self._is_visible(item.relative_to(path)) or self.load_hidden:
                try:
                    with open(item, 'rb') as f:
                        cal = Calendar.from_ical(f.read())
                        # Process the iCalendar events as desired
                        for component in cal.walk():
                            if component.name == 'VEVENT':

                                # Extract event properties and create a document object
                                summary = component.get('summary')
                                description = component.get('description')
                                start_time = datetime.fromisoformat(component.get('dtstart').dt.isoformat())
                                end_time = component.get('dtend')
                                if end_time is not None:
                                    end_time = datetime.fromisoformat(end_time.dt.isoformat())
                                event = Event(
                                    summary=summary,
                                    description=description,
                                    start_time=start_time,
                                    end_time=end_time,
                                    metadata = {'source': 'google calendar'}
                                )

                                # Create a document object and append it to the list
                                docs.append(event)
                except Exception as e:
                    self.logger.warning(e)
                    if self.silent_errors:
                        self.logger.warning(e)
                    else:
                        raise e
                finally:
                    if pbar:
                        pbar.update(1)
