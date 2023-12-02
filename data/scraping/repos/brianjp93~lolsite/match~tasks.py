"""match/tasks.py
"""
from django.conf import settings
from django.db.models.functions import Concat
from django.db.utils import IntegrityError
from django.db.models import Count, Subquery, OuterRef, Value
from django.db.models import Case, When, Sum
from django.db.models import IntegerField, Q, F
from django.utils import timezone
from django.db import connections, transaction
from pydantic import ValidationError
from data.constants import ARENA_QUEUE

from match.parsers.spectate import SpectateModel
from match.serializers import LlmMatchSerializer

from .parsers.match import BanType, MatchResponseModel, ParticipantModel, TeamModel
from .parsers.timeline import TimelineResponseModel
from .parsers import timeline as tmparsers

from .models import Match, MatchSummary, Participant, Stats
from .models import Team, Ban

from .models import AdvancedTimeline, Frame, ParticipantFrame
from .models import WardKillEvent, WardPlacedEvent
from .models import LevelUpEvent, SkillLevelUpEvent
from .models import ItemPurchasedEvent, ItemDestroyedEvent, ItemSoldEvent
from .models import ItemUndoEvent, TurretPlateDestroyedEvent
from .models import EliteMonsterKillEvent, ChampionSpecialKillEvent
from .models import BuildingKillEvent, GameEndEvent
from .models import ChampionKillEvent
from .models import VictimDamageDealt, VictimDamageReceived

from .models import Spectate

from lolsite.tasks import get_riot_api
from lolsite.helpers import query_debugger

from player.models import Summoner, NameChange
from player import tasks as pt

from lolsite.celery import app
import logging
import time
import json
from datetime import timedelta

from functools import partial
from typing import Optional

from multiprocessing.pool import ThreadPool

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


ROLES = ["top", "jg", "mid", "adc", "sup"]
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    pass


# @query_debugger
@app.task(name='match.tasks.import_match')
def import_match(match_id, region, refresh=False):
    """Import a match by its ID.

    Parameters
    ----------
    match_id : ID
    region : str
    refresh : bool
        Whether or not to re-import the match if it already exists.

    Returns
    -------
    None

    """
    api = get_riot_api()
    if api:
        r = api.match.get(match_id, region=region)
        match = r.content

        if r.status_code == 429:
            return "throttled"
        if r.status_code == 404:
            return "not found"

        import_match_from_data(match, region, refresh=refresh)


def fetch_match_json(match_id: str,  region: str, refresh=False):
    api = get_riot_api()
    r = api.match.get(match_id, region=region)
    match = r.content
    if r.status_code == 429:
        return "throttled"
    if r.status_code == 404:
        return "not found"
    return match


def import_summoner_from_participant(participants: list[ParticipantModel], region):
    sums = []
    for part in participants:
        if part.summonerId:
            summoner = Summoner(
                _id=part.summonerId,
                name=part.summonerName.strip(),
                simple_name=part.simple_name,
                region=region.lower(),
                puuid=part.puuid,
                riot_id_name=part.riotIdGameName,
                riot_id_tagline=part.riotIdTagline,
            )
            sums.append(summoner)
    Summoner.objects.bulk_create(sums, ignore_conflicts=True)


@app.task(name="match.tasks.handle_name_changes")
def handle_name_changes(days=30):
    """Create NameChange objects from Participant Data."""
    qs = Participant.objects.all().annotate(
        current_name=Subquery(
            Summoner.objects.filter(puuid=OuterRef('puuid')).values('simple_riot_id')[:1]
        )
    ).exclude(
        Q(current_name__iexact=Concat(
            F("riot_id_name"),
            Value("#"),
            F("riot_id_tagline"),
        ))
        | Q(current_name="")
    )
    if days:
        starts_at = timezone.now() - timedelta(days=days)
        timestamp = starts_at.timestamp() * 1000
        qs = qs.filter(match__game_creation__gt=timestamp)
    for participant in qs:
        try:
            summoner = Summoner.objects.filter(
                puuid=participant.puuid,
            ).values('id').get()
        except Summoner.DoesNotExist:
            return

        try:
            _, created = NameChange.objects.get_or_create(summoner_id=summoner['id'], old_name=participant.summoner_name)
            if created:
                logger.info(f"Created NameChange. Old Name = {participant.summoner_name}")
        except NameChange.MultipleObjectsReturned:
            qs = NameChange.objects.filter(
                summoner_id=summoner['id'],
                old_name=participant.summoner_name,
            ).order_by('created_date')
            for nc in qs[1:]:
                nc.delete()


def full_import(name=None, puuid=None, region=None, **kwargs):
    """Import only unimported games for a summoner.

    Looks at summoner.full_import_count to determine how many
    matches need to be imported.

    Parameters
    ----------
    name : str
    region : str
    season_id : ID
    puuid : ID
    queue : int
    beginTime : Epoch in ms
    endTime : Epoch in ms

    Returns
    -------
    None

    """
    if region is None:
        raise Exception("region parameter is required.")
    if name is not None:
        summoner_id = pt.import_summoner(region, name=name)
        summoner = Summoner.objects.get(id=summoner_id, region=region)
        puuid = summoner.puuid
    elif puuid is not None:
        summoner = Summoner.objects.get(puuid=puuid, region=region)
    else:
        raise Exception("name or puuid must be provided.")

    old_import_count = summoner.full_import_count
    # TODO: implement get_total_matches
    # total = get_total_matches(puuid, region, **kwargs)
    total = 100

    new_import_count = total - old_import_count
    if new_import_count > 0:
        logger.info(f"Importing {new_import_count} matches for {summoner.name}.")
        is_finished = import_recent_matches(0, new_import_count, puuid, region)
        if is_finished:
            summoner.full_import_count = total
            summoner.save()


def ranked_import(name=None, puuid=None, region=None, **kwargs):
    """Same as full_import except it only pulls from the 3 ranked queues.

    Parameters
    ----------
    name : str
    puuid : ID
    region : str
    season_id : ID
    puuid : ID
        the encrypted account ID
    queue : int
    beginTime : Epoch in ms
    endTime : Epoch in ms

    Returns
    -------
    None

    """
    kwargs["queue"] = [420, 440, 470]

    if region is None:
        raise Exception("region parameter is required.")
    if name is not None:
        summoner_id = pt.import_summoner(region, name=name)
        summoner = Summoner.objects.get(id=summoner_id, region=region)
        puuid = summoner.puuid
    elif puuid is not None:
        summoner = Summoner.objects.get(puuid=puuid, region=region)
    else:
        raise Exception("name or puuid must be provided.")

    old_import_count = summoner.ranked_import_count
    # TODO
    # total = get_total_matches(account_id, region, **kwargs)
    total = 100

    new_import_count = total - old_import_count
    if new_import_count > 0:
        logger.info(f"Importing {new_import_count} ranked matches for {summoner.name}.")
        is_finished = import_recent_matches(
            0, new_import_count, puuid, region, **kwargs
        )
        if is_finished:
            summoner.ranked_import_count = total
            summoner.save()


def pool_match_import(match_id: str, region: str, close_connections=True):
    match_json = fetch_match_json(match_id, region)
    import_match_from_data(match_json, region)
    if close_connections:
        connections.close_all()


@app.task(name="match.tasks.import_recent_matches")
def import_recent_matches(
    start: int,
    end: int,
    puuid: str,
    region: str,
    queue: Optional[int] = None,
    startTime: Optional[timezone.datetime] = None,
    endTime: Optional[timezone.datetime] = None,
):
    """Import recent matches for a puuid.

    Parameters
    ----------
    start : int
    end : int
    season_id : ID
    puuid : ID
        the encrypted account ID
    queue : int
    startTime : Epoch in ms
    endTime : Epoch in ms

    Returns
    -------
    int

    """
    has_more = True
    api = get_riot_api()
    import_count = 0
    if api:
        index = start
        size = 100
        if index + size > end:
            size = end - start
        please_continue = True
        while has_more and please_continue:
            riot_match_request_time = time.time()

            apicall = partial(
                api.match.filter,
                puuid,
                region=region,
                start=index,
                count=size,
                startTime=startTime,
                endTime=endTime,
                queue=queue,
            )
            r = apicall()
            logger.info('response: %s' % str(r))
            riot_match_request_time = time.time() - riot_match_request_time
            logger.info(
                f"Riot API match filter request time : {riot_match_request_time}"
            )
            try:
                if r.status_code == 404:
                    matches = []
                else:
                    matches = r.json()
            except Exception:
                time.sleep(10)
                r = apicall()
                if r.status_code == 404:
                    matches = []
                else:
                    matches = r.json()
            if len(matches) > 0:
                existing_ids = [x._id for x in Match.objects.filter(_id__in=matches)]
                new_matches = list(set(matches) - set(existing_ids))
                import_count += len(new_matches)
                jobs = [(x, region) for x in new_matches]
                with ThreadPool(processes=10) as pool:
                    start_time = time.perf_counter()
                    pool.starmap(pool_match_import, jobs)
                    logger.info(f'ThreadPool match import: {time.perf_counter() - start_time}')
            else:
                has_more = False
            index += size
            if index >= end:
                please_continue = False
    return import_count


@app.task(name="match.tasks.import_matches_for_popular_accounts")
def import_matches_for_popular_accounts(n=100):
    now = timezone.now()
    week = (now - timedelta(days=7)).date()
    qs = Summoner.objects.all().annotate(
        views=Sum('pageview__views', filter=Q(pageview__bucket_date__gte=week))
    ).order_by('-views').filter(views__isnull=False, views__gt=1)
    for i, summoner in enumerate(qs[:n]):
        logger.info(f"Importing matches for {summoner.name} ({i}) with {summoner.views=}")  # type: ignore
        try:
            import_recent_matches(0, 50, summoner.puuid, summoner.region)
        except Exception:
            logger.exception("Error while importing matches.")


@app.task(name="match.tasks.bulk_import")
def bulk_import(puuid: str, last_import_time_hours: int = 24, count=200, offset=10):
    now = timezone.now()
    thresh = now - timedelta(hours=last_import_time_hours)
    summoner: Summoner = Summoner.objects.get(puuid=puuid)
    if summoner.last_summoner_page_import is None or summoner.last_summoner_page_import < thresh:
        logger.info(f"Doing summoner page import for {summoner} of {count} games.")
        summoner.last_summoner_page_import = now
        summoner.save()
        import_recent_matches(offset, offset + count, puuid, region=summoner.region)


def get_top_played_with(
    summoner_id,
    team=True,
    season_id=None,
    queue_id=None,
    recent=None,
    recent_days=None,
    group_by="summoner_name",
):
    """Find the summoner names that you have played with the most.

    Parameters
    ----------
    summoner_id : int
        The *internal* Summoner ID
    team : bool
        Only count players who were on the same team
    season_id : int
    queue_id : int
    recent : int
        count of most recent games to check
    recent_days : int

    Returns
    -------
    query of counts

    """
    summoner = Summoner.objects.get(id=summoner_id)

    p = Participant.objects.all()
    if season_id is not None:
        p = p.filter(match__season_id=season_id)
    if queue_id is not None:
        p = p.filter(match__queue_id=queue_id)

    if recent is not None:
        m = Match.objects.all()
        if season_id is not None:
            m = m.filter(season_id=season_id)
        if queue_id is not None:
            m = m.filter(queue_id=queue_id)
        m = m.order_by("-game_creation")
        m_id_list = [x.id for x in m[:recent]]

        p = p.filter(match__id__in=m_id_list)
    elif recent_days is not None:
        now = timezone.now()
        start_time = now - timedelta(days=recent_days)
        start_time = int(start_time.timestamp() * 1000)
        p = p.filter(match__game_creation__gt=start_time)

    # get all participants that were in a match with the given summoner
    p = p.filter(match__participants__puuid=summoner.puuid)

    # exclude the summoner
    p = p.exclude(puuid=summoner.puuid)

    # I could include and `if team` condition, but I am assuming the top
    # values will be the same as the totals
    if not team:
        p = p.exclude(
            team_id=Subquery(
                Participant.objects.filter(
                    match__participants__id=OuterRef("id"),
                    puuid=summoner.puuid,
                ).values("team_id")[:1]
            )
        )
    else:
        p = p.filter(
            team_id=Subquery(
                Participant.objects.filter(
                    match__participants__id=OuterRef("id"),
                    puuid=summoner.puuid,
                ).values("team_id")[:1]
            )
        )
    p = p.annotate(
        win=Case(When(stats__win=True, then=1), default=0, output_field=IntegerField())
    )

    p = p.values(group_by).annotate(count=Count(group_by), wins=Sum("win"))
    p = p.order_by("-count")

    return p


@app.task(name="match.tasks.import_advanced_timeline")
def import_advanced_timeline(match_id: str, overwrite=False):
    victim_damage_received_events: list[VictimDamageReceived] = []
    victim_damage_dealt_events: list[VictimDamageDealt] = []
    ward_placed_events: list[WardPlacedEvent] = []
    ward_kill_events: list[WardKillEvent] = []
    item_purchase_events: list[ItemPurchasedEvent] = []
    item_destroyed_events: list[ItemDestroyedEvent] = []
    item_sold_events: list[ItemSoldEvent] = []
    item_undo_events: list[ItemUndoEvent] = []
    skill_level_up_events: list[SkillLevelUpEvent] = []
    level_up_events: list[LevelUpEvent] = []
    champion_special_kill_events: list[ChampionSpecialKillEvent] = []
    turret_plate_destroyed_events: list[TurretPlateDestroyedEvent] = []
    elite_monster_kill_events: list[EliteMonsterKillEvent] = []
    building_kill_events: list[BuildingKillEvent] = []
    with transaction.atomic():
        match = Match.objects.select_related('advancedtimeline').get(id=match_id)
        if hasattr(match, 'advancedtimeline') and overwrite:
            assert match.advancedtimeline
            match.advancedtimeline.delete()
        api = get_riot_api()
        region = match.platform_id.lower()
        logger.info(f"Requesting info for match {match.id} in region {region}")
        try:
            response = api.match.timeline(match._id, region=region)
            start = time.perf_counter()
            parsed = TimelineResponseModel.model_validate_json(response.content)
            logger.info(f"AdvancedTimeline parsing took: {time.perf_counter() - start}")
        except ValidationError:
            logger.exception('AdvanceTimeline could not be parsed.')
            return
        logger.info('Parsed AdvancedTimeline successfully.')
        data = parsed.info
        at = AdvancedTimeline(match=match, frame_interval=data.frameInterval)
        start_writing = time.perf_counter()
        at.save()

        for fm in data.frames:
            frame = Frame(timeline=at, timestamp=fm.timestamp)
            frame.save()
            pframes = []
            for pfm in fm.participantFrames.values():
                stats = pfm.championStats
                dmg_stats = pfm.damageStats
                p_frame_data = {
                    "frame": frame,

                    "participant_id": pfm.participantId,
                    "current_gold": pfm.currentGold,
                    "jungle_minions_killed": pfm.jungleMinionsKilled,
                    "gold_per_second": pfm.goldPerSecond,
                    "level": pfm.level,
                    "minions_killed": pfm.minionsKilled,
                    "time_enemy_spent_controlled": pfm.timeEnemySpentControlled,
                    "total_gold": pfm.totalGold,
                    "xp": pfm.xp,
                    "x": pfm.position.x,
                    "y": pfm.position.y,
                    "ability_haste": stats.abilityHaste,
                    "ability_power": stats.abilityPower,
                    "armor": stats.armor,
                    "armor_pen": stats.armorPen,
                    "armor_pen_percent": stats.armorPenPercent,
                    "attack_damage": stats.attackDamage,
                    "attack_speed": stats.attackSpeed,
                    "bonus_armor_pen_percent": stats.bonusArmorPenPercent,
                    "bonus_magic_pen_percent": stats.bonusMagicPenPercent,
                    "cc_reduction": stats.ccReduction,
                    "cooldown_reduction": stats.cooldownReduction,
                    "health": stats.health,
                    "health_max": stats.healthMax,
                    "health_regen": stats.healthRegen,
                    "lifesteal": stats.lifesteal,
                    "magic_pen": stats.magicPen,
                    "magic_pen_percent": stats.magicPenPercent,
                    "magic_resist": stats.magicResist,
                    "movement_speed": stats.movementSpeed,
                    "omnivamp": stats.omnivamp,
                    "physical_vamp": stats.physicalVamp,
                    "power": stats.power,
                    "power_max": stats.powerMax,
                    "power_regen": stats.powerRegen,
                    "spell_vamp": stats.spellVamp,
                    "magic_damage_done": dmg_stats.magicDamageDone,
                    "magic_damage_done_to_champions": dmg_stats.magicDamageDoneToChampions,
                    "magic_damage_taken": dmg_stats.magicDamageTaken,
                    "physical_damage_done": dmg_stats.physicalDamageDone,
                    "physical_damage_done_to_champions": dmg_stats.physicalDamageDoneToChampions,
                    "physical_damage_taken": dmg_stats.physicalDamageTaken,
                    "total_damage_done": dmg_stats.totalDamageDone,
                    "total_damage_done_to_champions": dmg_stats.totalDamageDoneToChampions,
                    "total_damage_taken": dmg_stats.totalDamageTaken,
                    "true_damage_done": dmg_stats.trueDamageDone,
                    "true_damage_done_to_champions": dmg_stats.trueDamageDoneToChampions,
                    "true_damage_taken": dmg_stats.trueDamageTaken,
                }
                pframes.append(ParticipantFrame(**p_frame_data))
            ParticipantFrame.objects.bulk_create(pframes)

            for evm in fm.events:
                assert frame.id
                match evm:
                    case tmparsers.WardPlacedEventModel():
                        ward_placed_events.append(WardPlacedEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            creator_id=evm.creatorId,
                            ward_type=evm.wardType,
                        ))
                    case tmparsers.WardKillEventModel():
                        ward_kill_events.append(WardKillEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            killer_id=evm.killerId,
                            ward_type=evm.wardType,
                        ))
                    case tmparsers.PauseEndEventModel():
                        ...
                    case tmparsers.PauseStartEventModel():
                        ...
                    case tmparsers.ObjectiveBountyPrestartEventModel():
                        ...
                    case tmparsers.ObjectiveBountyFinishEventModel():
                        ...
                    case tmparsers.ChampionTransformEventModel():
                        ...
                    case tmparsers.DragonSoulGivenEventModel():
                        ...
                    case tmparsers.ItemPurchasedEventModel():
                        item_purchase_events.append(ItemPurchasedEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            item_id=evm.itemId,
                            participant_id=evm.participantId,
                        ))
                    case tmparsers.ItemDestroyedEventModel():
                        item_destroyed_events.append(ItemDestroyedEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            item_id=evm.itemId,
                            participant_id=evm.participantId,
                        ))
                    case tmparsers.ItemSoldEventModel():
                        item_sold_events.append(ItemSoldEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            item_id=evm.itemId,
                            participant_id=evm.participantId,
                        ))
                    case tmparsers.ItemUndoEventModel():
                        item_undo_events.append(ItemUndoEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            participant_id=evm.participantId,
                            before_id=evm.beforeId,
                            after_id=evm.afterId,
                            gold_gain=evm.goldGain,
                        ))
                    case tmparsers.SkillLevelUpEventModel():
                        skill_level_up_events.append(SkillLevelUpEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            level_up_type=evm.levelUpType,
                            participant_id=evm.participantId,
                            skill_slot=evm.skillSlot,
                        ))
                    case tmparsers.LevelUpModel():
                        level_up_events.append(LevelUpEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            level=evm.level,
                            participant_id=evm.participantId,
                        ))
                    case tmparsers.ChampionSpecialKillEventModel():
                        champion_special_kill_events.append(ChampionSpecialKillEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            assisting_participant_ids=evm.assistingParticipantIds,
                            kill_type=evm.killType,
                            killer_id=evm.killerId,
                            multi_kill_length=evm.multiKillLength,
                            x=evm.position.x,
                            y=evm.position.y,
                        ))
                    case tmparsers.TurretPlateDestroyedEventModel():
                        turret_plate_destroyed_events.append(TurretPlateDestroyedEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            killer_id=evm.killerId,
                            lane_type=evm.laneType,
                            x=evm.position.x,
                            y=evm.position.y,
                            team_id=evm.teamId,
                        ))
                    case tmparsers.EliteMonsterKillEventModel():
                        elite_monster_kill_events.append(EliteMonsterKillEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            killer_id=evm.killerId,
                            killer_team_id=evm.killerTeamId,
                            bounty=evm.bounty,
                            x=evm.position.x,
                            y=evm.position.y,
                            monster_type=evm.monsterType,
                            monster_sub_type=evm.monsterSubType,
                        ))
                    case tmparsers.BuildingKillEventModel():
                        building_kill_events.append(BuildingKillEvent(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            killer_id=evm.killerId,
                            bounty=evm.bounty,
                            x=evm.position.x,
                            y=evm.position.y,
                            building_type=evm.buildingType,
                            lane_type=evm.laneType,
                            tower_type=evm.towerType,
                            team_id=evm.teamId,
                        ))
                    case tmparsers.GameEndEventModel():
                        GameEndEvent.objects.create(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            game_id=evm.gameId,
                            real_timestamp=evm.realTimestamp,
                            winning_team=evm.winningTeam,
                        )
                    case tmparsers.ChampionKillEventModel():
                        cke = ChampionKillEvent.objects.create(
                            frame_id=frame.id,
                            timestamp=evm.timestamp,
                            bounty=evm.bounty,
                            shutdown_bounty=evm.shutdownBounty,
                            kill_streak_length=evm.killStreakLength,
                            killer_id=evm.killerId,
                            victim_id=evm.victimId,
                            x=evm.position.x,
                            y=evm.position.y,
                        )
                        for vd in evm.victimDamageDealt or []:
                            assert cke.id
                            victim_damage_dealt_events.append(VictimDamageDealt(
                                championkillevent_id=cke.id,
                                basic=vd.basic,
                                magic_damage=vd.magicDamage,
                                name=vd.name,
                                participant_id=vd.participantId,
                                physical_damage=vd.physicalDamage,
                                spell_name=vd.spellName,
                                spell_slot=vd.spellSlot,
                                true_damage=vd.trueDamage,
                                type=vd.type,
                            ))
                        for vd in evm.victimDamageReceived or []:
                            assert cke.id
                            victim_damage_received_events.append(VictimDamageReceived(
                                championkillevent_id=cke.id,
                                basic=vd.basic,
                                magic_damage=vd.magicDamage,
                                name=vd.name,
                                participant_id=vd.participantId,
                                physical_damage=vd.physicalDamage,
                                spell_name=vd.spellName,
                                spell_slot=vd.spellSlot,
                                true_damage=vd.trueDamage,
                                type=vd.type,
                            ))
        WardPlacedEvent.objects.bulk_create(ward_placed_events)
        WardKillEvent.objects.bulk_create(ward_kill_events)
        ItemPurchasedEvent.objects.bulk_create(item_purchase_events)
        ItemDestroyedEvent.objects.bulk_create(item_destroyed_events)
        ItemSoldEvent.objects.bulk_create(item_sold_events)
        ItemUndoEvent.objects.bulk_create(item_undo_events)
        SkillLevelUpEvent.objects.bulk_create(skill_level_up_events)
        LevelUpEvent.objects.bulk_create(level_up_events)
        ChampionSpecialKillEvent.objects.bulk_create(champion_special_kill_events)
        TurretPlateDestroyedEvent.objects.bulk_create(turret_plate_destroyed_events)
        EliteMonsterKillEvent.objects.bulk_create(elite_monster_kill_events)
        BuildingKillEvent.objects.bulk_create(building_kill_events)
        VictimDamageDealt.objects.bulk_create(victim_damage_dealt_events)
        VictimDamageReceived.objects.bulk_create(victim_damage_received_events)
        end_writing = time.perf_counter()
        logger.info(f"Writing Advanced Timeline took {end_writing - start_writing}.")


def import_spectate_from_data(parsed: SpectateModel, region: str):
    spectate_data = {
        "game_id": parsed.gameId,
        "region": region,
        "platform_id": parsed.platformId,
        "encryption_key": parsed.observers.encryptionKey,
    }
    spectate = Spectate(**spectate_data)
    try:
        spectate.save()
    except IntegrityError:
        # already saved
        pass


def import_summoners_from_spectate(parsed: SpectateModel, region):
    summoners = {}
    for part in parsed.participants:
        if part.summonerId:
            sum_data = {
                "name": part.summonerName.strip(),
                "region": region,
                "profile_icon_id": part.profileIconId,
                "_id": part.summonerId,
            }
            summoner = Summoner(**sum_data)
            try:
                summoner.save()
                summoners[summoner._id] = summoner
            except IntegrityError:
                try:
                    summoner = Summoner.objects.get(region=region, _id=part.summonerId)
                    summoners[summoner._id] = summoner
                except Summoner.DoesNotExist:
                    pass
    return summoners


def get_player_ranks(summoner_list, threshold_days=1, sync=True):
    logger.info('Applying player ranks.')
    jobs = [pt.import_positions.s(x.id, threshold_days=threshold_days) for x in summoner_list]
    jobs = [(x.id, threshold_days) for x in summoner_list]
    if jobs:
        if sync:
            for x in jobs:
                pt.import_positions(*x)
        else:
            with ThreadPool(processes=10) as pool:
                def pool_position_import(a, b):
                    pt.import_positions(a, b)
                    connections.close_all()
                start_time = time.perf_counter()
                pool.starmap(pool_position_import, jobs)
                logger.info(f'ThreadPool positions import: {time.perf_counter() - start_time}')


def apply_player_ranks(match, threshold_days=1):
    if not isinstance(match, Match):
        match = Match.objects.get(id=match)

    now = timezone.now()
    one_day_ago = now - timedelta(days=1)
    if match.get_creation() > one_day_ago:
        # ok -- apply ranks
        parts = match.participants.all()
        q = Q()
        for part in parts:
            q |= Q(_id=part.summoner_id, puuid=part.puuid)
        summoner_qs = Summoner.objects.filter(q)
        summoner_list = [x for x in summoner_qs]
        summoners = {x.puuid: x for x in summoner_qs}
        get_player_ranks(summoner_list, threshold_days=threshold_days, sync=False)

        to_save = []
        for part in parts:
            if not part.tier:
                # only applying if it is not already applied
                summoner = summoners.get(part.puuid)
                if summoner:
                    checkpoint = summoner.get_newest_rank_checkpoint()
                    if checkpoint:
                        query = checkpoint.positions.filter(
                            queue_type="RANKED_SOLO_5x5"
                        )
                        if query:
                            position = query[0]
                            part.rank, part.tier = position.rank, position.tier
                            to_save.append(part)
            else:
                # if any tiers are already applied, stop
                return
        Participant.objects.bulk_update(to_save, fields=['rank', 'tier'])


PARTICIPANT_ROLE_KEYS = {
    "TOP": 0,
    "JUNGLE": 5,
    "MIDDLE": 10,
    "BOTTOM": 15,
    "UTILITY": 20,
}


def participant_key(participant: Participant):
    """Use riot's `team_position` variable and order from top to sup."""
    return (
        participant.team_id,
        PARTICIPANT_ROLE_KEYS.get(participant.team_position, 25),
    )


def get_sorted_participants(match: Match, participants=None):
    if not participants:
        participants = match.participants.all().select_related("stats")
    if len(participants) == 10:
        ordered = sorted(list(participants), key=participant_key)
    elif match.queue_id == ARENA_QUEUE:
        ordered = sorted(list(participants), key=lambda x: x.placement)
    else:
        ordered = list(participants)
    return ordered


def build_participant(part: ParticipantModel, match: Match):
    return Participant(
        match=match,
        _id=part.participantId,
        summoner_id=part.summonerId,
        puuid=part.puuid,
        summoner_name=part.summonerName,
        summoner_name_simplified=part.simple_name,
        champion_id=part.championId,
        champ_experience=part.champExperience,
        summoner_1_id=part.summoner1Id,
        summoner_1_casts=part.summoner1Casts,
        summoner_2_id=part.summoner2Id,
        summoner_2_casts=part.summoner2Casts,
        team_id=part.teamId,
        lane=part.lane,
        role=part.role,
        individual_position=part.individualPosition,
        team_position=part.teamPosition,
        placement=part.placement,
        subteam_placement=part.subteamPlacement,
        riot_id_name=part.riotIdGameName,
        riot_id_tagline=part.riotIdTagline,
    )

def build_team(team: TeamModel, match: Match):
    return Team(
        match=match,
        _id=team.teamId,
        baron_kills=team.objectives.baron.kills,
        first_baron=team.objectives.baron.first,
        dragon_kills=team.objectives.dragon.kills,
        first_dragon=team.objectives.dragon.first,
        first_blood=team.objectives.champion.first,
        first_inhibitor=team.objectives.inhibitor.first,
        inhibitor_kills=team.objectives.inhibitor.kills,
        first_rift_herald=team.objectives.riftHerald.first,
        rift_herald_kills=team.objectives.riftHerald.kills,
        first_tower=team.objectives.tower.first,
        tower_kills=team.objectives.tower.kills,
        win=team.win,
    )

def build_ban(ban: BanType, team: Team):
    return Ban(
        champion_id=ban.championId,
        pick_turn=ban.pickTurn,
        team=team,
    )

def build_stats(part: ParticipantModel):
    return Stats(
        # participant=participant_model,

        all_in_pings=part.allInPings,
        assist_me_pings=part.assistMePings,
        bait_pings=part.baitPings,
        basic_pings=part.basicPings,
        command_pings=part.commandPings,
        danger_pings=part.dangerPings,
        enemy_missing_pings=part.enemyMissingPings,
        enemy_vision_pings=part.enemyVisionPings,
        get_back_pings=part.getBackPings,
        hold_pings=part.holdPings,
        need_vision_pings=part.needVisionPings,
        on_my_way_pings=part.onMyWayPings,
        push_pings=part.pushPings,
        vision_cleared_pings=part.visionClearedPings,

        game_ended_in_early_surrender=part.gameEndedInEarlySurrender,
        game_ended_in_surrender=part.gameEndedInSurrender,
        riot_id_name=part.riotIdGameName,
        riot_id_tagline=part.riotIdTagline,

        assists=part.assists,
        champ_level=part.champLevel,
        damage_dealt_to_objectives=part.damageDealtToObjectives,
        damage_dealt_to_turrets=part.damageDealtToTurrets,
        damage_self_mitigated=part.damageSelfMitigated,
        deaths=part.deaths,
        double_kills=part.doubleKills,
        first_blood_assist=part.firstBloodAssist,
        first_blood_kill=part.firstBloodKill,
        first_tower_assist=part.firstTowerAssist,
        first_tower_kill=part.firstTowerKill,
        gold_earned=part.goldEarned,
        inhibitor_kills=part.inhibitorKills,
        item_0=part.item0,
        item_1=part.item1,
        item_2=part.item2,
        item_3=part.item3,
        item_4=part.item4,
        item_5=part.item5,
        item_6=part.item6,
        killing_sprees=part.killingSprees,
        kills=part.kills,
        largest_critical_strike=part.largestCriticalStrike,
        largest_killing_spree=part.largestKillingSpree,
        largest_multi_kill=part.largestMultiKill,
        longest_time_spent_living=part.longestTimeSpentLiving,
        magic_damage_dealt=part.magicDamageDealt,
        magic_damage_dealt_to_champions=part.magicDamageDealtToChampions,
        magical_damage_taken=part.magicDamageTaken,
        neutral_minions_killed=part.neutralMinionsKilled,
        penta_kills=part.pentaKills,
        stat_perk_0=part.stat_perk_0,
        stat_perk_1=part.stat_perk_1,
        stat_perk_2=part.stat_perk_2,
        perk_0=part.perks.primary_style.selections[0].perk,
        perk_1=part.perks.primary_style.selections[1].perk,
        perk_2=part.perks.primary_style.selections[2].perk,
        perk_3=part.perks.primary_style.selections[3].perk,
        perk_4=part.perks.sub_style.selections[0].perk,
        perk_5=part.perks.sub_style.selections[1].perk,

        perk_0_var_1=part.perks.primary_style.selections[0].var1,
        perk_1_var_1=part.perks.primary_style.selections[1].var1,
        perk_2_var_1=part.perks.primary_style.selections[2].var1,
        perk_3_var_1=part.perks.primary_style.selections[3].var1,
        perk_4_var_1=part.perks.sub_style.selections[0].var1,
        perk_5_var_1=part.perks.sub_style.selections[1].var1,

        perk_0_var_2=part.perks.primary_style.selections[0].var2,
        perk_1_var_2=part.perks.primary_style.selections[1].var2,
        perk_2_var_2=part.perks.primary_style.selections[2].var2,
        perk_3_var_2=part.perks.primary_style.selections[3].var2,
        perk_4_var_2=part.perks.sub_style.selections[0].var2,
        perk_5_var_2=part.perks.sub_style.selections[1].var2,

        perk_0_var_3=part.perks.primary_style.selections[0].var3,
        perk_1_var_3=part.perks.primary_style.selections[1].var3,
        perk_2_var_3=part.perks.primary_style.selections[2].var3,
        perk_3_var_3=part.perks.primary_style.selections[3].var3,
        perk_4_var_3=part.perks.sub_style.selections[0].var3,
        perk_5_var_3=part.perks.sub_style.selections[1].var3,

        perk_primary_style=part.perks.primary_style.style,
        perk_sub_style=part.perks.sub_style.style,
        spell_1_casts=part.spell1Casts,
        spell_2_casts=part.spell2Casts,
        spell_3_casts=part.spell3Casts,
        spell_4_casts=part.spell4Casts,
        time_ccing_others=part.timeCCingOthers,
        total_damage_dealt=part.totalDamageDealt,
        total_damage_dealt_to_champions=part.totalDamageDealtToChampions,
        total_damage_taken=part.totalDamageTaken,
        total_damage_shielded_on_teammates=part.totalDamageShieldedOnTeammates,
        total_heal=part.totalHeal,
        total_heals_on_teammates=part.totalHealsOnTeammates,
        total_minions_killed=part.totalMinionsKilled,
        total_time_crowd_control_dealt=part.totalTimeCCDealt,
        total_units_healed=part.totalUnitsHealed,
        total_ally_jungle_minions_killed=part.totalAllyJungleMinionsKilled or 0,
        total_enemy_jungle_minions_killed=part.totalEnemyJungleMinionsKilled or 0,
        triple_kills=part.tripleKills,
        true_damage_dealt=part.trueDamageDealt,
        true_damage_dealt_to_champions=part.trueDamageDealtToChampions,
        true_damage_taken=part.trueDamageTaken,
        turret_kills=part.turretKills,
        unreal_kills=part.unrealKills,
        vision_score=part.visionScore,
        vision_wards_bought_in_game=part.visionWardsBoughtInGame,
        wards_killed=part.wardsKilled,
        wards_placed=part.wardsPlaced,
        win=part.win,
    )

@transaction.atomic()
def import_match_from_data(data, region: str, refresh=False):
    try:
        parsed = MatchResponseModel.model_validate_json(data)
    except ValidationError:
        logger.exception('Match could not be parsed.')
        return

    if "tutorial" in parsed.info.gameMode.lower():
        return False

    if parsed.info.gameDuration == 0:
        return False

    info = parsed.info
    sem_ver = info.sem_ver
    match_model = Match(
        _id=parsed.metadata.matchId,
        game_creation=info.gameCreation,
        game_duration=info.gameDuration,
        game_mode=info.gameMode,
        map_id=info.mapId,
        platform_id=info.platformId,
        queue_id=info.queueId,
        game_version=info.gameVersion,
        major=sem_ver.get(0, ''),
        minor=sem_ver.get(1, ''),
        patch=sem_ver.get(2, ''),
        build=sem_ver.get(3, ''),
        is_fully_imported=True,
    )
    try:
        match_model.save()
    except IntegrityError:
        if refresh:
            Match.objects.filter(_id=parsed.metadata.matchId).delete()
            match_model.save()
        else:
            logging.warning("Attempted to import game which was already imported. Ignoring.")
            return

    participants_data = info.participants
    import_summoner_from_participant(participants_data, region)

    participants: list[Participant] = []
    stats: list[Stats] = []
    for part in participants_data:
        # PARTICIPANT
        participant_model = build_participant(part, match_model)
        participants.append(participant_model)

        # STATS
        stats_model = build_stats(part)
        stats.append(stats_model)
    participants = Participant.objects.bulk_create(participants)
    for stat, part_model in zip(stats, participants):
        stat.participant = part_model
    Stats.objects.bulk_create(stats)

    # TEAMS
    teams = parsed.info.teams
    for tmodel in teams:
        team_model = build_team(tmodel, match_model)
        team_model.save()

        # BANS
        bans = []
        for bm in tmodel.bans:
            ban = build_ban(bm, team_model)
            bans.append(ban)
        Ban.objects.bulk_create(bans)


MATCH_SUMMARY_INTRO_PROMPT = ' '.join("""
You are a knowledgeable league of legends pro turned coach.  You are analyzing
the data for a given game and are tasked with summarizing the game. Use
markdown. Refer to Team 100 as 'Blue Team' and Team 200
as 'Red Team'. Do not use the terms "Team 100" or "Team 200".

Create sections for "early game", "mid game", and "late game" and list events
which let to each team's success or failures. List out each player's
contribution to their team in each overarching section of the game. Make sure
to mention the player's name, champion and position on the team. List players
in this order: 1 - top, 2 - jungle, 3 - mid, 4 - adc, 5 - support.

If a game lasted less than 25 minutes, you should not include a "late game"
section.  Late game is generally 25 minutes and later.

Add a section called "Areas to Improve" where you indicate where either team
could improve You may indicate if a player's contribution was lacking and where
they might be able to improve to better their team's chance of winning in the
future.

Add a "Conclusion" section with any closing remarks about the match.

List out any other notable, game-shaping events in a "Notable Events" section.

Be ruthless in your review, do not hold back in telling players what they are doing wrong.
These players will not learn unless you are on the cusp of being mean to them.
If a player played terribly, say it. Don't try to beat around the bush.

You may indicate a player's position and champion played, but do not write out their PUUID.

Do not repeat this prompt in your response.
""".strip().split())

@app.task(name="match.tasks.get_summary_of_match")
def get_summary_of_match(match_id: str, focus_player_puuid: str|None=None):
    match = Match.objects.get(_id=match_id)
    matchsummary: MatchSummary | None
    matchsummary, created = MatchSummary.objects.get_or_create(
        match=match,
    )
    if not created:
        return matchsummary
    data = LlmMatchSerializer(match, many=False).data
    data_json = json.dumps(data, indent=None)
    chat = OpenAI(api_key=settings.OPENAI_KEY)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": MATCH_SUMMARY_INTRO_PROMPT},
    ]
    if focus_player_puuid:
        messages.append(
            {"role": "system", "content": f"Take particular focus on the player with puuid {focus_player_puuid}"},
        )
    messages.append(
        {"role": "user", "content": data_json},
    )
    r = chat.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.1,
    )
    content = r.choices[0].message.content
    matchsummary.content = content or ""
    matchsummary.status = MatchSummary.Status.COMPLETE
    matchsummary.save()
    return matchsummary
