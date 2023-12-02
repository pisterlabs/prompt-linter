"""SQLite database to store messages."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from django.core.exceptions import ValidationError
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Model, QuerySet
from loguru import logger

from chatgpt.chatgpt import ChatGPT
from sqlitedb.models import (
    Conversation,
    CurrentConversation,
    User,
    UserConversations,
    UserImages,
)
from sqlitedb.utils import ErrorCodes

if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar("T", bound=Model)


class SQLiteDatabase(object):
    """SQLite database Object."""

    def get_user(self: Self, telegram_id: int) -> User | ErrorCodes:
        """Retrieve a User object from the database for a given user_id. If the user does not exist, create a new user.

        Args:
            telegram_id (int): The ID of the user to retrieve or create.

        Returns
        -------
            Union[User, int]: The User object corresponding to the specified user ID, or -1 if an error occurs.
        """
        try:
            user: User
            user, created = User.objects.get_or_create(
                telegram_id=telegram_id,
                defaults={"name": f"User {telegram_id}"},
            )
        except Exception as e:
            logger.error(
                f"Unable to get or create user: {e} because of {type(e).__name__}",
            )
            return ErrorCodes.exceptions
        else:
            if created:
                logger.info(f"Created new user {user}")
            else:
                logger.info(f"Retrieved existing {user}")
        return user

    def get_messages_by_user(self: Self, telegram_id: int) -> Any:
        """Retrieve a list of messages for a user from the database.

        Args:
            telegram_id (int): The ID of the user for which to retrieve messages.

        Returns
        -------
            Any: A list of message tuples for the specified user. Each tuple contains two elements:
                                    - from_bot: a boolean indicating whether the message is from the bot
                                    - message: the text of the message
        """
        logger.debug("Getting user messages")
        user = self.get_user(telegram_id)

        try:
            current_conversation: CurrentConversation = CurrentConversation.objects.get(
                user=user,
            )
        except CurrentConversation.DoesNotExist:
            logger.error(f"No current conversation found for user with ID {user}.")
            return []

        messages = (
            UserConversations.objects.select_related("conversation")
            .filter(user=user, conversation=current_conversation.conversation)
            .values("from_bot", "message")
        )
        return messages.order_by("message_date")

    def _get_current_conversation(
        self: Self,
        user: User,
        message: str,
    ) -> ErrorCodes | int:
        """Retrieve a CurrentConversation object from the database given a User object.

        Args:
            user (User): The User object for which to retrieve the current conversation.

        Returns
        -------
            int: The ID of the user's current conversation, or -1 if an error occurs.
        """
        logger.debug(f"Getting current conversation for user {user}")
        try:
            current_conversation = CurrentConversation.objects.get(user=user)
            conversation_id = int(current_conversation.conversation_id)
        except CurrentConversation.DoesNotExist:
            logger.info(f"No current conversation exists for user {user}")
            try:
                openai_response = ChatGPT.send_text_completion_request(message)
                if isinstance(openai_response, ErrorCodes):
                    logger.debug("Unable to get chat title from OpenAI.")
                    return ErrorCodes.exceptions
                chat_title = openai_response["choices"][0]["text"]
                conversation = Conversation(user=user, title=chat_title)
                conversation.save()
                current_conversation = CurrentConversation(
                    user=user,
                    conversation=conversation,
                )
                current_conversation.save()
                conversation_id = int(current_conversation.conversation_id)
            except Exception as e:
                logger.error(f"Unable to create new conversation {e}")
                return ErrorCodes.exceptions
        return conversation_id

    def _create_conversation(
        self: Self,
        user_id: int,
        message: str,
        from_bot: bool,
    ) -> ErrorCodes | None:
        """Create a new Conversations object and save it to the database.

        Args:
            user_id (int): The ID of the user who sent the message.
            message (str): The message text to be saved in the Conversations object.
            from_bot (bool): Whether the message is from the bot.

        Returns
        -------
            int: 0 if the conversation is successfully created and saved, or -1 if an error occurs.
        """
        try:
            user = self.get_user(user_id)
            if isinstance(user, User):
                conversation_id = self._get_current_conversation(user, message)
                if isinstance(conversation_id, ErrorCodes):
                    logger.error("Unable to get current conversation")
                    return ErrorCodes.exceptions
                conversation = UserConversations(
                    user=user,
                    message=message,
                    from_bot=from_bot,
                    conversation_id=conversation_id,
                )
                conversation.save()
                return None
            logger.debug("Unable to get user")
            return ErrorCodes.exceptions
        except Exception as e:
            logger.error(f"Unable to save conversation {e}")
            return ErrorCodes.exceptions

    def insert_message_from_user(
        self: Self,
        message: str,
        user_id: int,
    ) -> ErrorCodes | None:
        """Insert a new conversation message into the database for a user.

        Args:
            message (str): The message text to be saved in the Conversations object.
            user_id (int): The ID of the user who sent the message.

        Returns
        -------
            int: 0 if the conversation is successfully created and saved, or -1 if an error occurs.
        """
        logger.debug("Inserting message from user.")
        return self._create_conversation(user_id, message, False)

    def insert_message_from_gpt(
        self: Self,
        message: str,
        user_id: int,
    ) -> ErrorCodes | None:
        """Insert a new conversation message into the database from the GPT model.

        Args:
            message (str): The message text to be saved in the Conversations object.
            user_id (int): The ID of the user who received the message from the GPT model.

        Returns
        -------
            int: 0 if the conversation is successfully created and saved, or -1 if an error occurs.
        """
        return self._create_conversation(user_id, message, True)

    def insert_images_from_gpt(
        self: Self,
        image_caption: str,
        image_url: str,
        telegram_id: int,
    ) -> ErrorCodes | None:
        """Insert a new image record into the database for a user.

        Args:
            image_caption (str): The caption text for the image (optional).
            image_url (str): The URL of the image file.
            telegram_id (int): The ID of the user who uploaded the image.

        Returns
        -------
            int: 0 if the image is successfully created and saved, or -1 if an error occurs.
        """
        try:
            user = self.get_user(telegram_id)
            image = UserImages(
                user=user,
                image_caption=image_caption,
                image_url=image_url,
                from_bot=True,
            )
            image.save()
        except Exception as e:
            logger.error(f"Unable to save image {e}")
            return ErrorCodes.exceptions

    def delete_all_user_messages(self: Self, telegram_id: int) -> ErrorCodes | int:
        """Delete all conversations for a user from the database.

        Args:
            telegram_id (int): The ID of the user for which to delete conversations.

        Returns
        -------
        int: The number of conversations deleted, or an error code if an error occurs.
        """
        try:
            user = self.get_user(telegram_id)
            conversations = Conversation.objects.filter(user=user)
            return int(conversations.delete()[0])
        except Exception as e:
            logger.error(f"Error deleting {e}")
            return ErrorCodes.exceptions

    def delete_all_user_images(self: Self, telegram_id: int) -> ErrorCodes | int:
        """Delete all images for a user from the database.

        Args:
            telegram_id (int): The ID of the user for which to delete images.

        Returns
        -------
        int: The number of images deleted, or an error code if an error occurs.
        """
        try:
            user = self.get_user(telegram_id)
            images = UserImages.objects.filter(user=user)
            return int(images.delete()[0])
        except Exception as e:
            logger.error(f"Error deleting {e}")
            return ErrorCodes.exceptions

    def delete_all_user_data(
        self: Self,
        telegram_id: int,
    ) -> tuple[ErrorCodes | int, ErrorCodes | int]:
        """Delete all conversations and images for a user from the database.

        Args:
            telegram_id (int): The ID of the user for which to delete data.

        Returns
        -------
            Tuple[int, int]: A tuple containing the number of conversations and images that were deleted, respectively,
             or (-1,-1) if an error occurs.
        """
        num_conv_deleted, num_img_deleted = self.delete_all_user_messages(
            telegram_id,
        ), self.delete_all_user_images(telegram_id)
        return num_conv_deleted, num_img_deleted

    def initiate_new_conversation(
        self: Self,
        telegram_id: int,
        title: str,
    ) -> ErrorCodes | None:
        """Initiates a new conversation for a user by creating a new Conversation object and a new CurrentConversation
        object.

        Args:
            telegram_id (int): The ID of the user for which to start a new conversation.
            title (str): The title of the new conversation.

        Returns
        -------
            int: The result of the conversation creation operation.
        """
        user = self.get_user(telegram_id)

        conversation = Conversation(user=user, title=title)
        try:
            conversation.save()
        except ValidationError as ve:
            logger.error(
                f"Validation error while saving conversation to the database: {ve}",
            )
            return ErrorCodes.exceptions
        except Exception as e:
            logger.error(
                f"An error occurred while saving conversation to the database: {e}",
            )
            return ErrorCodes.exceptions

        try:
            current_conversation, created = CurrentConversation.objects.get_or_create(
                user=user,
                defaults={"conversation": conversation},
            )
            if not created:
                current_conversation.conversation = conversation
                current_conversation.save()
            return None
        except ValidationError as ve:
            logger.error(
                f"Validation error while getting or creating current conversation: {ve}",
            )
            conversation.delete()
            return ErrorCodes.exceptions

        except Exception as e:
            logger.error(
                f"An error occurred while getting or creating current conversation: {e}",
            )
            conversation.delete()
            return ErrorCodes.exceptions

    def initiate_empty_new_conversation(self: Self, telegram_id: int) -> None:
        """Initiates a new empty conversation for a user by creating a new Conversation object and a new
        CurrentConversation object.

        Args:
            telegram_id (int): The ID of the user for which to start a new conversation.

        Returns
        -------
            ConversationResult: The result of the conversation creation operation.
        """
        user = self.get_user(telegram_id)

        try:
            CurrentConversation.objects.get(user=user).delete()
        except CurrentConversation.DoesNotExist:
            logger.info(f"No current conversation for user {user}")

    def _paginate_queryset(
        self: Self,
        queryset: QuerySet[T],
        page: int,
        per_page: int,
    ) -> dict[str, Any]:
        """Helper function to paginate a given queryset.

        Args:
            queryset (QuerySet[T]): The queryset to be paginated.
            page (int): The current page number.
            per_page (int): The number of items to display per page.

        Returns
        -------
            dict: A dictionary containing the paginated data and pagination details.
        """
        paginator = Paginator(queryset, per_page)

        try:
            paginated_data = paginator.page(page)
        except PageNotAnInteger:
            logger.debug(f"{page} is not a valid page number")
            paginated_data = paginator.page(1)
        except EmptyPage:
            logger.debug(f"Empty {page} current page")
            paginated_data = paginator.page(paginator.num_pages)
        logger.info(f"Got {len(paginated_data)} records")
        return {
            "data": paginated_data,
            "total_data": paginator.count,
            "total_pages": paginator.num_pages,
            "current_page": paginated_data.number,
            "has_previous": paginated_data.has_previous(),
            "has_next": paginated_data.has_next(),
        }

    def get_user_conversations(self: Self, telegram_id: int, page: int, per_page: int) -> Any:
        """Return a paginated list of conversations for a given user.

        Args:
            telegram_id (int): The ID of the user.
            page (int): The current page number.
            per_page (int): The number of conversations to display per page.

        Returns
        -------
            dict: A dictionary containing the paginated conversations and pagination details.
        """
        user = self.get_user(telegram_id)

        # Retrieve the conversations for the given user
        conversations = Conversation.objects.only("id", "title", "start_time").filter(user=user).order_by("-start_time")

        # Use the helper function to paginate the queryset
        return self._paginate_queryset(conversations, page, per_page)

    def get_conversation(
        self: Self,
        conversation_id: int,
        user: User,
    ) -> Conversation | None:
        """Get the Conversation object by its ID.

        Args:
            conversation_id (int): The ID of the conversation.
            user (User): User to get the Conversation

        Returns
        -------
            Optional[Conversation]: The conversation object if it exists, otherwise None.
        """
        try:
            conversation = Conversation.objects.only("id", "title").get(
                id=conversation_id,
                user=user,
            )
            logger.debug(f"Got {conversation}")
            return conversation
        except Conversation.DoesNotExist:
            return None

    def set_active_conversation(self: Self, user: User, conversation: Conversation) -> None:
        """Set the active conversation for a user.

        Args:
            user (User): The user whose active conversation is to be set.
            conversation (Conversation): The conversation to be set as active.
            Pass None to unset the active conversation.
        """
        current_conversation, created = CurrentConversation.objects.get_or_create(
            user=user,
        )
        current_conversation.conversation = conversation
        current_conversation.save()

    def get_conversation_messages(
        self: Self,
        conversation_id: int,
        telegram_id: int,
        page: int,
        per_page: int,
    ) -> Any:
        """Return a paginated list of conversations for a given user.

        Args:
            conversation_id (int): Conversation ID.
            telegram_id (int): The ID of the user.
            page (int): The current page number.
            per_page (int): The number of conversations to display per page.

        Returns
        -------
            dict: A dictionary containing the paginated conversations and pagination details.
        """
        user = self.get_user(telegram_id)

        # Retrieve the conversations for the given user
        messages = (
            UserConversations.objects.only("message", "from_bot")
            .filter(user=user, conversation_id=conversation_id)
            .order_by("message_date")
        )

        # Use the helper function to paginate the queryset
        return self._paginate_queryset(messages, page, per_page)
