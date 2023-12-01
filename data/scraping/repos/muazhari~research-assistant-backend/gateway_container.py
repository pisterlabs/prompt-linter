from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer

from app.outers.containers.setting_container import SettingContainer
from app.outers.interfaces.gateways.clients.openai_client import OpenAiClient


class GatewayContainer(DeclarativeContainer):
    settings: SettingContainer = providers.DependenciesContainer()

    openai_client: OpenAiClient = providers.Singleton(
        OpenAiClient,
        openai_setting=settings.openai
    )
