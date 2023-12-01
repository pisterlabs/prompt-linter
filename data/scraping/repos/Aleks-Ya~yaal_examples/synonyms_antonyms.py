# Anki add-on.
# 1. Adds "Fill synonyms and antonyms" item in "Main menu"/"Tools".
# 2. On click, perform find by regex and replace in specified note fields.
import logging
from csv import DictReader
from io import StringIO
from typing import Sequence, List

from anki.collection import OpChanges, OpChangesWithCount, Collection
from anki.notes import NoteId, Note
from aqt import mw
from aqt.operations import CollectionOp, ResultWithChanges
from aqt.qt import QAction, qconnect
from aqt.utils import showInfo
from openai.types.chat import ChatCompletion

from fields import english_field, part_of_speech_field, synonym1_field, synonyms_field, antonyms_field
from tags import absent_synonym1_tag, absent_synonyms_tag, absent_antonyms_tag, unit_tag
from . import openai_client

log: logging.Logger = logging.getLogger(__name__)

dry_run: bool = False


class WantCancelException(Exception):
    pass


def _update_notes(notes_to_update: List[Note], col: Collection, changes: OpChangesWithCount):
    log.info(f"Notes to update in slice: {len(notes_to_update)}")
    lines: List[str] = []
    for note in notes_to_update:
        word: str = note[english_field]
        if not note[part_of_speech_field]:
            raise RuntimeError(f"Part of speech is missing: nid={note.id}")
        pos: str = note[part_of_speech_field]
        line: str = f'"{note.id}","{word}","{pos}"'
        lines.append(line)
    lines_str: str = '\n'.join(lines)
    nid_column: str = 'ID'
    english_column: str = 'English Word'
    pos_column: str = 'Part Of Speech'
    synonym_number: int = 5
    antonym_number: int = 5
    synonym_headers: List[str] = [f'Synonym {i}' for i in range(1, synonym_number + 1)]
    antonym_headers: List[str] = [f'Antonym {i}' for i in range(1, antonym_number + 1)]
    log.debug(f"synonym_headers: {synonym_headers}")
    log.debug(f"antonym_headers: {antonym_headers}")
    synonym_headers_quoted: List[str] = [f'"{synonym_header}"' for synonym_header in synonym_headers]
    antonym_headers_quoted: List[str] = [f'"{antonym_header}"' for antonym_header in antonym_headers]
    log.debug(f"synonym_headers_quoted: {synonym_headers_quoted}")
    log.debug(f"antonym_headers_quoted: {synonym_headers_quoted}")
    all_headers_quoted: List[str] = list(synonym_headers_quoted)
    all_headers_quoted += antonym_headers_quoted
    prompt: str = (
        f'I will provide you a CSV snippet. '
        f'You need to fill columns {", ".join(all_headers_quoted)} in the snippet. '
        f'Use the most simple and wide-spread synonyms/antonyms if possible. '
        f'One-word synonyms/antonyms are preferred over multi-words. '
        f'Your response must contain strictly only raw CSV content (it is very important). '
        f'The CSV snippet is:\n'
        f'```\n'
        f'"{nid_column}","{english_column}","{pos_column}",{",".join(all_headers_quoted)}\n'
        f'{lines_str}\n'
        f'```\n'
    )
    log.debug(f"Prompt:\n{prompt}")
    chat_completion: ChatCompletion = openai_client.get_completion(prompt)
    message: str = chat_completion.choices[0].message.content
    log.debug(f"Message:\n{message}")
    message = message.replace('```\n', '').replace('\n```', '')
    log.debug(f"Message without MarkDown:\n{message}")
    with StringIO(message) as csv_file:
        reader: DictReader = DictReader(csv_file, doublequote=True)
        for row in reader:
            if mw.progress.want_cancel():
                raise WantCancelException()
            log.debug(f"Row: {row}")
            nid_int: int = int(row[nid_column])
            note: Note = col.get_note(NoteId(nid_int))
            synonym1_old: str = note[synonym1_field]
            synonyms_old: str = note[synonyms_field]
            antonyms_old: str = note[antonyms_field]
            is_synonym1_empty: bool = synonym1_old.strip() == ''
            is_synonyms_empty: bool = synonyms_old.strip() == ''
            is_antonyms_empty: bool = antonyms_old.strip() == ''
            if note[english_field] != row[english_column]:
                raise RuntimeError(f"Wrong English word: note={note[english_field]}, row={row[english_column]}")
            if not is_synonym1_empty and not is_synonyms_empty and not is_antonyms_empty:
                raise RuntimeError(f"Fields {synonym1_field}, {synonyms_field} and {antonyms_field} are all not empty: "
                                   f"nid={note.id}, synonym1='{synonym1_old}', synonyms='{synonyms_old}', "
                                   f"antonyms='{antonyms_old}'")
            full_synonyms: List[str] = [row[synonym_header] for synonym_header in synonym_headers
                                        if row[synonym_header]]
            full_antonyms: List[str] = [row[antonym_header] for antonym_header in antonym_headers
                                        if row[antonym_header]]
            if is_synonym1_empty:
                synonym1_new: str = full_synonyms[0] if len(full_synonyms) > 0 else ""
                if is_synonyms_empty:
                    if synonym1_new in full_synonyms:
                        full_synonyms.remove(synonym1_new)
                    synonyms_new: str = ", ".join(full_synonyms)
                else:
                    synonyms_new: str = synonyms_old
            else:
                synonym1_new: str = synonym1_old
                if is_synonyms_empty:
                    if synonym1_new in full_synonyms:
                        full_synonyms.remove(synonym1_new)
                    synonyms_new: str = ", ".join(full_synonyms)
                else:
                    synonyms_new: str = synonyms_old
            antonyms_new: str = ", ".join(full_antonyms) if is_antonyms_empty else antonyms_old
            note[synonym1_field] = synonym1_new
            note[synonyms_field] = synonyms_new
            note[antonyms_field] = antonyms_new

            tags_old: List[str] = list(note.tags)
            if synonym1_new == '':
                note.tags.append(absent_synonym1_tag)
            else:
                if absent_synonym1_tag in tags_old:
                    note.tags.remove(absent_synonym1_tag)
            if synonyms_new == '':
                note.tags.append(absent_synonyms_tag)
            else:
                if absent_synonyms_tag in tags_old:
                    note.tags.remove(absent_synonyms_tag)
            if antonyms_new == '':
                note.tags.append(absent_antonyms_tag)
            else:
                if absent_antonyms_tag in tags_old:
                    note.tags.remove(absent_antonyms_tag)

            log.info(f"Updating note: nid={note.id}, english='{note[english_field]}', "
                     f"pos='{note[part_of_speech_field]}', "
                     f"synonym1='{synonym1_old}'->'{synonym1_new}', "
                     f"synonyms='{synonyms_old}'->'{synonyms_new}', "
                     f"antonyms='{antonyms_old}'->'{antonyms_new}', "
                     f"tags='{tags_old}'->'{note.tags}'")
            if not dry_run:
                col.update_note(note)
            else:
                log.warning(f"Skipped update because dry_run={dry_run}")
            changes.count += 1


def _background_operation(col: Collection) -> ResultWithChanges:
    changes: OpChangesWithCount = OpChangesWithCount()
    query: str = (f'(({synonym1_field}: -tag:{absent_synonym1_tag}) OR '
                  f'({synonyms_field}: -tag:{absent_synonyms_tag}) OR '
                  f'({antonyms_field}: -tag:{absent_antonyms_tag})) '
                  f'-tag:{unit_tag}')
    log.info(f"Query: '{query}'")
    note_ids: Sequence[NoteId] = col.find_notes(query)
    log.info(f"Found notes: {len(note_ids)}")
    notes_to_update: List[Note] = []
    for note_id in note_ids:
        note: Note = col.get_note(note_id)
        synonym1_old: str = note[synonym1_field]
        synonyms_old: str = note[synonyms_field]
        antonyms_old: str = note[antonyms_field]
        need_update: bool = not synonym1_old or not synonyms_old or not antonyms_old
        if need_update:
            notes_to_update.append(note)
    notes_to_update_size: int = len(notes_to_update)
    log.info(f"Notes to update: {notes_to_update_size}")
    _show_progress(changes, notes_to_update_size)
    slice_size: int = 10

    for i in range(0, notes_to_update_size, slice_size):
        log.info(f"Processed notes: {i} from {notes_to_update_size}")
        sublist: List[Note] = notes_to_update[i:i + slice_size]
        try:
            _update_notes(sublist, col, changes)
            _show_progress(changes, notes_to_update_size)
        except WantCancelException:
            log.info("Cancelling")
            return changes
    return changes


def _show_progress(changes: OpChangesWithCount, notes_len: int):
    mw.taskman.run_on_main(
        lambda: mw.progress.update(
            label=f"Updated: {changes.count} from {notes_len}",
            value=changes.count,
            max=notes_len
        )
    )


def on_success(changes: ResultWithChanges) -> None:
    dry_run_str: str = " (DRY RUN)" if dry_run else ""
    showInfo(f"Finished: updated {changes.count} notes{dry_run_str}")


def on_failure(e: Exception) -> None:
    showInfo(f"Failed: {e}")


def _ui_action():
    op: CollectionOp[OpChanges] = CollectionOp(parent=mw, op=lambda col: _background_operation(col))
    op.success(on_success)
    op.failure(on_failure)
    op.run_in_background()


def menu_action() -> QAction:
    action = QAction('Fill synonyms and antonyms', mw)
    qconnect(action.triggered, _ui_action)
    return action
