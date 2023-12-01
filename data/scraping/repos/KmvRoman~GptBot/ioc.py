from aiogram import Bot
from aiohttp import ClientSession
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.access_context.use_case import AccessContext
from src.application.available_user.use_case import AvailableUser
from src.application.check_follow_to_channel.use_case import CheckToFollow
from src.application.message_with_context.use_case import CreateMessageWithContext
from src.application.message_without_context.use_case import CreateMessageWithoutContext
from src.application.messenger_message_reader.use_case import MessengerMessageToMessage
from src.application.remote_limit.use_case import CreateLimit
from src.application.send_message.use_case import SendMessage
from src.application.set_rate.use_case import SetRate
from src.application.user_create.use_case import CreateUser
from src.domain.context.service.access import ContextAccessService
from src.domain.context.service.context import ContextService
from src.domain.user.services.access import SubscriptionAccessService
from src.domain.user.services.subscription import SubscriptionService
from src.domain.user.services.user import UserService
from src.infrastructure.adapters.database.repositories.user_repository import UserRepository
from src.infrastructure.adapters.messenger_adapter.send_message import MessengerAdapter
from src.infrastructure.adapters.openai.openai import OpenAiAdapter
from src.infrastructure.config.config import Config
from src.infrastructure.ioc.interfaces import InteractorFactory
from src.infrastructure.openai.chat_completion.chat_completion import RequestToOpenAi


class IoC(InteractorFactory):
    def __init__(self, repo: UserRepository, bot: Bot, aiohttp_session: ClientSession, config: Config):
        self.db_gateway = repo
        self.user_service = UserService()
        self.subscription_service = SubscriptionService()
        self.subscription_access_service = SubscriptionAccessService()
        self.context_access_service = ContextAccessService()
        self.context_service = ContextService()
        self.bot = MessengerAdapter(bot=bot)
        openai_session = RequestToOpenAi(
            session=aiohttp_session, api_key=config.openai.key,
            chat_completion=config.openai.chat_completion_url,
        )
        self.openai = OpenAiAdapter(openai=openai_session)

    async def commit(self):
        await self.db_gateway.commit()

    async def check_available(self) -> AvailableUser:
        return AvailableUser(db_gateway=self.db_gateway, user_service=self.user_service)

    async def create_user(self) -> CreateUser:
        return CreateUser(
            db_gateway=self.db_gateway,
            user_service=self.user_service,
            subscription_service=self.subscription_service,
        )

    async def check_follow_to_channel(self) -> CheckToFollow:
        return CheckToFollow(db_gateway=self.db_gateway, messenger_gateway=self.bot)

    async def messenger_message_reader(self) -> MessengerMessageToMessage:
        return MessengerMessageToMessage(db_gateway=self.db_gateway)

    async def remote_limit(self) -> CreateLimit:
        return CreateLimit(
            db_gateway=self.db_gateway,
            access_send_service=self.subscription_access_service,
            subscription_service=self.subscription_service,
        )

    async def send_message(self) -> SendMessage:
        return SendMessage(
            db_gateway=self.db_gateway,
            messenger_gateway=self.bot,
            context_service=self.context_service,
        )

    async def message_without_context(self) -> CreateMessageWithoutContext:
        return CreateMessageWithoutContext(
            db_gateway=self.db_gateway,
            request_to_api=self.openai,
            context_service=self.context_service,
        )

    async def message_with_context(self) -> CreateMessageWithContext:
        return CreateMessageWithContext(
            db_gateway=self.db_gateway,
            request_to_api=self.openai,
            context_service=self.context_service,
        )

    async def access_context(self) -> AccessContext:
        return AccessContext(db_gateway=self.db_gateway, context_access_service=self.context_access_service)

    async def set_rate(self) -> SetRate:
        return SetRate(db_gateway=self.db_gateway, subscription_service=self.subscription_service)
