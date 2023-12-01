"""Define the utility for archiving data."""
from uuid import UUID
import boto3
from loguru import logger
from langchain.schema import HumanMessage
# first imports are for local development, second imports are for deployment
try:
    from .archive_schemas import BaseArchiveRecord, HumanMessageRecord
    from ..shared_schemas import DateRange
    from ..taitutors.llm_schemas import (
        BaseMessage,
    )
except ImportError:
    from taibackend.databases.archive_schemas import BaseArchiveRecord, HumanMessageRecord
    from taibackend.shared_schemas import DateRange
    from taibackend.taitutors.llm_schemas import (
        BaseMessage,
    )


class Archive:
    """Define the utility for archiving data."""
    def __init__(self, bucket_name: str) -> None:
        """Instantiate the utility for archiving data."""
        self._bucket_name = bucket_name
        self._bucket = boto3.resource('s3').Bucket(self._bucket_name)

    def archive_message(self, message: BaseMessage, class_id: UUID) -> None:
        """Store the message."""
        base_record = BaseArchiveRecord(
            class_id=class_id,
            timestamp=message.timestamp,
        )
        if isinstance(message, HumanMessage):
            archive_record = HumanMessageRecord(
                message=message.content,
                **base_record.dict(),
            )
        else:
            logger.warning(f"Archive does not support archiving messages of type {message.__class__.__name__}")
        self.put_archive_record(archive_record)

    def get_archived_messages(self, class_id: UUID, date_range: DateRange, RecordClass: BaseArchiveRecord) -> list[BaseArchiveRecord]:
        """Get archived messages for a class."""
        prefix = RecordClass.get_archive_prefix(class_id)
        objects = self._bucket.objects.filter(Prefix=prefix)
        archive_records = []
        for obj in objects:
            # load the object as it's in json format
            archive_record = RecordClass.parse_raw(obj.get()['Body'].read())
            if date_range.start_date <= archive_record.timestamp <= date_range.end_date:
                archive_records.append(archive_record)
        return archive_records

    def put_archive_record(self, archive_record: BaseArchiveRecord) -> None:
        """Put the archive record in the archive."""
        self._bucket.put_object(
            Key=archive_record.get_archive_object_key(),
            Body=archive_record.json(),
        )
