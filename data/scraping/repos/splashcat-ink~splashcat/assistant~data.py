import json
from io import StringIO, BytesIO

from django.conf import settings
from django.db.models import Prefetch
from openai import OpenAI

from battles.models import Player, Battle, BattleGroup

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def upload_user_battles_to_openai(user):
    player_query = Player.objects.select_related(
        "title_adjective__string",
        "title_subject__string",
        "nameplate_background",
        "nameplate_badge_1__description",
        "nameplate_badge_2__description",
        "nameplate_badge_3__description",
        "weapon__name",
        "weapon__sub__name",
        "weapon__special__name",
        "head_gear__gear__name",
        "head_gear__primary_ability__name",
        "head_gear__secondary_ability_1__name",
        "head_gear__secondary_ability_2__name",
        "head_gear__secondary_ability_3__name",
        "clothing_gear__gear__name",
        "clothing_gear__primary_ability__name",
        "clothing_gear__secondary_ability_1__name",
        "clothing_gear__secondary_ability_2__name",
        "clothing_gear__secondary_ability_3__name",
        "shoes_gear__gear__name",
        "shoes_gear__primary_ability__name",
        "shoes_gear__secondary_ability_1__name",
        "shoes_gear__secondary_ability_2__name",
        "shoes_gear__secondary_ability_3__name",
    )
    player_prefetch = Prefetch(
        'teams__players',
        queryset=player_query,
    )
    battles = user.battles.select_related("vs_stage__name").prefetch_related("awards__name",
                                                                             player_prefetch).order_by(
        '-played_time')
    battle_array = []
    battle: Battle
    for battle in battles:
        battle_data = battle.to_gpt_dict()
        battle_array.append(json.dumps(battle_data) + '\n')
    temp_file = StringIO("")
    temp_file.writelines(battle_array)
    temp_file.seek(0)
    temp_file = BytesIO(temp_file.read().encode('utf-8'))
    temp_file.seek(0)
    openai_file = client.files.create(
        file=temp_file,
        purpose='assistants'
    )
    return openai_file


def upload_battle_to_openai(battle: Battle):
    battle_array = [json.dumps(battle.to_gpt_dict()) + '\n']
    temp_file = StringIO("")
    temp_file.writelines(battle_array)
    temp_file.seek(0)
    temp_file = BytesIO(temp_file.read().encode('utf-8'))
    temp_file.seek(0)
    openai_file = client.files.create(
        file=temp_file,
        purpose='assistants'
    )
    return openai_file


def upload_battle_group_to_openai(battle_group: BattleGroup):
    player_query = Player.objects.select_related(
        "title_adjective__string",
        "title_subject__string",
        "nameplate_background",
        "nameplate_badge_1__description",
        "nameplate_badge_2__description",
        "nameplate_badge_3__description",
        "weapon__name",
        "weapon__sub__name",
        "weapon__special__name",
        "head_gear__gear__name",
        "head_gear__primary_ability__name",
        "head_gear__secondary_ability_1__name",
        "head_gear__secondary_ability_2__name",
        "head_gear__secondary_ability_3__name",
        "clothing_gear__gear__name",
        "clothing_gear__primary_ability__name",
        "clothing_gear__secondary_ability_1__name",
        "clothing_gear__secondary_ability_2__name",
        "clothing_gear__secondary_ability_3__name",
        "shoes_gear__gear__name",
        "shoes_gear__primary_ability__name",
        "shoes_gear__secondary_ability_1__name",
        "shoes_gear__secondary_ability_2__name",
        "shoes_gear__secondary_ability_3__name",
    )
    player_prefetch = Prefetch(
        'teams__players',
        queryset=player_query,
    )
    battles = battle_group.battles.select_related("vs_stage__name").prefetch_related("awards__name",
                                                                                     player_prefetch).order_by(
        '-played_time')
    battle_array = []
    battle: Battle
    for battle in battles:
        battle_data = battle.to_gpt_dict()
        battle_array.append(json.dumps(battle_data) + '\n')
    temp_file = StringIO("")
    temp_file.writelines(battle_array)
    temp_file.seek(0)
    temp_file = BytesIO(temp_file.read().encode('utf-8'))
    temp_file.seek(0)
    openai_file = client.files.create(
        file=temp_file,
        purpose='assistants'
    )
    return openai_file
