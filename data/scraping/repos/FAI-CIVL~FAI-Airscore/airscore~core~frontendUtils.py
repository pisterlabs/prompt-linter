import datetime
import json
from functools import partial
from os import environ, scandir
from pathlib import Path

import requests
import ruamel.yaml
from calcUtils import c_round, sec_to_time
from db.conn import db_session
from db.tables import (
    TblCompAuth,
    TblCompetition,
    TblRegion,
    TblRegionWaypoint,
    TblTask,
    TblTaskWaypoint,
)
from Defines import IGCPARSINGCONFIG, MAPOBJDIR, filename_formats, track_formats
from flask import current_app, jsonify
from map import make_map
from route import Turnpoint
from sqlalchemy import func
from sqlalchemy.orm import aliased


def create_menu(active: str = '') -> list:
    import Defines

    menu = [dict(title='Competitions', url='public.home', css='nav-item')]
    if Defines.LADDERS:
        menu.append(dict(title='Ladders', url='public.ladders', css='nav-item'))
    menu.append(dict(title='Pilots', url='public.pilots', css='nav-item'))
    if Defines.WAYPOINT_AIRSPACE_FILE_LIBRARY:
        menu.append(dict(title='Flying Areas', url='public.regions', css='nav-item'))

    return menu


def get_comps() -> list:
    c = aliased(TblCompetition)
    with db_session() as db:
        comps = (
            db.query(
                c.comp_id,
                c.comp_name,
                c.comp_site,
                c.comp_class,
                c.sanction,
                c.comp_type,
                c.date_from,
                c.date_to,
                func.count(TblTask.task_id).label('tasks'),
                c.external,
            )
            .outerjoin(TblTask, c.comp_id == TblTask.comp_id)
            .group_by(c.comp_id)
        )

    return [row._asdict() for row in comps]


def find_orphan_pilots(pilots_list: list, orphans: list) -> (list, list):
    """Tries to guess participants that do not have a pil_id, and associate them to other participants or
    to a pilot in database"""
    from calcUtils import get_int
    from db.tables import PilotView as P

    pilots_found = []
    still_orphans = []
    ''' find a match among pilots already in list'''
    print(f"trying to find pilots from orphans...")
    for p in orphans:
        name, civl_id, comp_id = p['name'], p['civl_id'], p['comp_id']
        found = next(
            (
                el
                for el in pilots_list
                if (el['name'] == name or (civl_id and civl_id == el['civl_id'])) and comp_id not in el[comp_id]
            ),
            None,
        )
        if found:
            '''adding to existing pilot'''
            found['par_ids'].append(p['par_id'])
            found['comp_ids'].append(comp_id)
        else:
            still_orphans.append(p)
    ''' find a match among pilots in database if still we have orphans'''
    orphans = []
    if still_orphans:
        with db_session() as db:
            pilots = db.query(P).all()
            for p in still_orphans:
                name, civl_id, comp_id = p['name'].title(), p['civl_id'], p['comp_id']
                row = next(
                    (
                        el
                        for el in pilots
                        if (
                            (
                                el.first_name
                                and el.last_name
                                and el.first_name.title() in name
                                and el.last_name.title() in name
                            )
                            or (civl_id and el.civl_id and civl_id == get_int(el.civl_id))
                        )
                    ),
                    None,
                )
                if row:
                    '''check if we already found the same pilot in orphans'''
                    found = next((el for el in pilots_found if el['pil_id'] == row.pil_id), None)
                    if found:
                        found['par_ids'].append(p['par_id'])
                        found['comp_ids'].append(comp_id)
                    else:
                        name = f"{row.first_name.title()} {row.last_name.title()}"
                        pilot = dict(
                            comp_ids=[p['comp_id']],
                            par_ids=[p['par_id']],
                            pil_id=int(row.pil_id),
                            civl_id=get_int(row.civl_id) or None,
                            fai_id=row.fai_id,
                            name=name,
                            sex=p['sex'],
                            nat=p['nat'],
                            glider=p['glider'],
                            glider_cert=p['glider_cert'],
                            results=[],
                        )
                        pilots_found.append(pilot)
                else:
                    orphans.append(p)
    pilots_list.extend(pilots_found)

    return pilots_list, orphans


def get_ladders() -> list:
    from db.tables import TblCountryCode as C
    from db.tables import TblLadder as L
    from db.tables import TblLadderSeason as LS

    with db_session() as db:
        ladders = (
            db.query(
                L.ladder_id, L.ladder_name, L.ladder_class, L.date_from, L.date_to, C.natIoc.label('nat'), LS.season
            )
            .join(LS, L.ladder_id == LS.ladder_id)
            .join(C, L.nation_code == C.natId)
            .filter(LS.active == 1)
            .order_by(LS.season.desc())
        )

    return [row._asdict() for row in ladders]


def get_ladder_results(
    ladder_id: int, season: int, nat: str = None, starts: datetime.date = None, ends: datetime.date = None
) -> json:
    """creates result json using comp results from all events in ladder"""
    import time

    from calcUtils import get_season_dates
    from compUtils import get_nat
    from db.tables import TblCompetition as C
    from db.tables import TblLadder as L
    from db.tables import TblLadderComp as LC
    from db.tables import TblLadderSeason as LS
    from db.tables import TblParticipant as P
    from db.tables import TblResultFile as R
    from result import open_json_file

    if not (nat and starts and ends):
        lad = L.get_by_id(ladder_id)
        nat_code, date_from, date_to = lad.nation_code, lad.date_from, lad.date_to
        nat = get_nat(nat_code)
        '''get season start and end day'''
        starts, ends = get_season_dates(ladder_id=ladder_id, season=season, date_from=date_from, date_to=date_to)
    with db_session() as db:
        '''get ladder info'''
        # probably we could keep this from ladder list page?
        row = (
            db.query(
                L.ladder_id, L.ladder_name, L.ladder_class, LS.season, LS.cat_id, LS.overall_validity, LS.validity_param
            )
            .join(LS)
            .filter(L.ladder_id == ladder_id, LS.season == season)
            .one()
        )
        rankings = create_classifications(row.cat_id)
        info = {
            'ladder_name': row.ladder_name,
            'season': row.season,
            'ladder_class': row.ladder_class,
            'id': row.ladder_id,
        }
        formula = {'overall_validity': row.overall_validity, 'validity_param': row.validity_param}

        '''get comps and files'''
        results = (
            db.query(C.comp_id, R.filename)
            .join(LC)
            .join(R, (R.comp_id == C.comp_id) & (R.task_id.is_(None)) & (R.active == 1))
            .filter(C.date_to > starts, C.date_to < ends, LC.c.ladder_id == ladder_id)
        )
        comps_ids = [row.comp_id for row in results]
        files = [row.filename for row in results]
        print(comps_ids, files)

        '''create Participants list'''
        results = db.query(P).filter(P.comp_id.in_(comps_ids), P.nat == nat).order_by(P.pil_id, P.comp_id).all()
        pilots_list = []
        orphans = []
        for row in results:
            if row.pil_id:
                p = next((el for el in pilots_list if el['pil_id'] == row.pil_id), None)
                if p:
                    '''add par_id'''
                    p['par_ids'].append(row.par_id)
                    p['comp_ids'].append(row.comp_id)
                else:
                    '''insert a new pilot'''
                    p = dict(
                        comp_ids=[row.comp_id],
                        par_ids=[row.par_id],
                        pil_id=row.pil_id,
                        civl_id=row.civl_id,
                        fai_id=row.fai_id,
                        name=row.name,
                        sex=row.sex,
                        nat=row.nat,
                        glider=row.glider,
                        glider_cert=row.glider_cert,
                        results=[],
                    )
                    pilots_list.append(p)
            else:
                p = dict(
                    comp_id=row.comp_id,
                    pil_id=row.pil_id,
                    par_id=row.par_id,
                    civl_id=row.civl_id,
                    fai_id=row.fai_id,
                    name=row.name,
                    sex=row.sex,
                    nat=row.nat,
                    glider=row.glider,
                    glider_cert=row.glider_cert,
                )
                orphans.append(p)
    '''try to guess orphans'''
    if orphans:
        pilots_list, orphans = find_orphan_pilots(pilots_list, orphans)

    '''get results'''
    stats = {'tot_pilots': len(pilots_list)}
    comps = []
    tasks = []
    for file in files:
        f = open_json_file(file)
        '''get comp info'''
        i = f['info']
        comp_code = i['comp_code']
        results = f['results']
        comps.append(dict(id=i['id'], comp_code=i['comp_code'], comp_name=i['comp_name'], tasks=len(f['tasks'])))
        tasks.extend(
            [
                dict(id=t['id'], ftv_validity=t['ftv_validity'], task_code=f"{i['comp_code']}_{t['task_code']}")
                for t in f['tasks']
            ]
        )
        for r in results:
            p = next((el for el in pilots_list if r['par_id'] in el['par_ids']), None)
            if p:
                scores = r['results']
                for i, s in scores.items():
                    idx, code = next((t['id'], t['task_code']) for t in tasks if f"{comp_code}_{i}" == t['task_code'])
                    p['results'].append({'task_id': idx, 'task_code': code, **s})

    '''get params'''
    val = formula['overall_validity']
    param = formula['validity_param']
    stats['valid_tasks'] = len(tasks)
    stats['total_validity'] = c_round(sum([t['ftv_validity'] for t in tasks]), 4)
    stats['avail_validity'] = (
        0
        if len(tasks) == 0
        else c_round(stats['total_validity'] * param, 4)
        if val == 'ftv'
        else stats['total_validity']
    )

    '''calculate scores'''
    for pil in pilots_list:
        dropped = 0 if not (val == 'round' and param) else int(len(pil['results']) / param)
        pil['score'] = 0

        '''reset scores in list'''
        for res in pil['results']:
            res['score'] = res['pre']

        ''' if we score all tasks, or tasks are not enough to have discards,
            or event has just one valid task regardless method,
            we can simply sum all score values
        '''
        if not ((val == 'all') or (val == 'round' and dropped == 0) or (len(tasks) < 2) or len(pil['results']) < 2):
            '''create a ordered list of results, score desc (perf desc if ftv)'''
            sorted_results = sorted(
                pil['results'], key=lambda x: (x['perf'], x['pre'] if val == 'ftv' else x['pre']), reverse=True
            )
            if val == 'round' and dropped:
                for i in range(1, dropped + 1):
                    sorted_results[-i]['score'] = 0  # getting id of worst result task
            elif val == 'ftv':
                '''ftv calculation'''
                pval = stats['avail_validity']
                for res in sorted_results:
                    if not (pval > 0):
                        res['score'] = 0
                    else:
                        '''get ftv_validity of corresponding task'''
                        tval = next(t['ftv_validity'] for t in tasks if t['task_code'] == res['task_code'])
                        if pval > tval:
                            '''we can use the whole score'''
                            pval -= tval
                        else:
                            '''we need to calculate proportion'''
                            res['score'] = c_round(res['score'] * (pval / tval))
                            pval = 0

            '''calculates final pilot score'''
            pil['results'] = sorted_results
            pil['score'] = sum(r['score'] for r in sorted_results)

    '''order results'''
    pilots_list = sorted(pilots_list, key=lambda x: x['score'], reverse=True)
    stats['winner_score'] = 0 if not pilots_list else pilots_list[0]['score']
    '''create json'''
    file_stats = {'timestamp': time.time()}
    output = {
        'info': info,
        'comps': comps,
        'formula': formula,
        'stats': stats,
        'results': pilots_list,
        'rankings': rankings,
        'file_stats': file_stats,
    }
    return output


def get_admin_comps(current_userid, current_user_access=None):
    """get a list of all competitions in the DB and flag ones where owner is current user"""
    c = aliased(TblCompetition)
    ca = aliased(TblCompAuth)
    with db_session() as db:
        comps = (
            db.query(
                c.comp_id,
                c.comp_name,
                c.comp_site,
                c.date_from,
                c.date_to,
                func.count(TblTask.task_id),
                c.external,
                ca.user_id,
            )
            .outerjoin(TblTask, c.comp_id == TblTask.comp_id)
            .outerjoin(ca)
            .filter(ca.user_auth == 'owner')
            .group_by(c.comp_id, ca.user_id)
        )
    all_comps = []
    for c in comps:
        comp = list(c)
        comp[1] = f'<a href="/users/comp_settings_admin/{comp[0]}">{comp[1]}</a>'
        comp[3] = comp[3].strftime("%Y-%m-%d")
        comp[4] = comp[4].strftime("%Y-%m-%d")
        comp[6] = 'Imported' if comp[6] else ''
        if (int(comp[7]) == current_userid) or (current_user_access in ('admin', 'manager')):
            comp[7] = 'delete'
        else:
            comp[7] = ''
        all_comps.append(comp)
    return jsonify({'data': all_comps})


def get_task_list(comp_id: int) -> dict:
    """returns a dict of tasks info"""
    from compUtils import get_tasks_details

    tasks = get_tasks_details(comp_id)
    max_task_num = 0
    last_region = 0
    for task in tasks:
        tasknum = task['task_num']
        if int(tasknum) > max_task_num:
            max_task_num = int(tasknum)
            last_region = task['reg_id']

        task['num'] = f"Task {tasknum}"
        task['opt_dist'] = 0 if not task['opt_dist'] else c_round(task['opt_dist'] / 1000, 2)
        task['opt_dist'] = f"{task['opt_dist']} km"
        if task['comment'] is None:
            task['comment'] = ''
        if not task['track_source']:
            task['track_source'] = ''
        task['date'] = task['date'].strftime('%d/%m/%y')

        task['needs_full_rescore'] = False
        task['needs_new_scoring'] = False
        task['needs_recheck'] = False
        task['ready_to_score'] = False

        if not (task['locked'] or task['cancelled']):
            '''check if task needs tracks recheck or rescoring'''
            task['needs_new_scoring'], task['needs_recheck'], task['needs_full_rescore'] = check_task(task['task_id'])
            '''check if we have all we need to be able to accept tracks and score'''
            task['ready_to_score'] = (
                                         task['opt_dist']
                                         and task['window_open_time']
                                         and task['window_close_time']
                                         and task['start_time']
                                         and task['start_close_time']
                                         and task['task_deadline']
                                     ) is not None

    return {'next_task': max_task_num + 1, 'last_region': last_region, 'tasks': tasks}


def switch_task_lock(task_id: int, old_value: bool) -> bool:
    """Locks a task (making results official) if it is open, and vice versa"""
    from db.tables import TblTask
    from task import get_task_json_filename
    from result import update_result_status, update_tasks_status_in_comp_result
    value = not old_value
    try:
        '''check task has a valid active result'''
        result = get_task_json_filename(task_id)
        if value and not result:
            '''cannot lock task'''
            return False
        task = TblTask.get_by_id(task_id)
        comp_ud = task.comp_id
        task.update(locked=value)
        '''change status'''
        status = 'Official Result' if value else 'Provisional Results'
        update_result_status(result, status=status, locked=value)
        update_tasks_status_in_comp_result(comp_ud)
        return True
    except Exception:
        return False


def switch_task_cancelled(task_id: int, old_value: bool, comment: str = None) -> bool:
    """Declares a task Cancelled (and locked) if it is active, and vice versa"""
    from db.tables import TblTask
    value = not old_value
    task = TblTask.get_by_id(task_id)
    task.update(cancelled=value, locked=value, comment=comment)
    return True


def check_task_turnpoints(task_id: int, wpt_id: int) -> dict:
    from task import Task, write_map_json
    task = Task.read(task_id)
    tps = task.turnpoints
    last_edited = next(tp for tp in tps if tp.wpt_id == wpt_id)
    edited = False
    '''check launch'''
    if not tps[0].type == 'launch':
        tps[0].type = 'launch'
        edited = True
    for tp in tps[1:]:
        if tp.type == 'launch':
            tp.type = 'waypoint'
            edited = True
        elif tp.type == 'speed' and last_edited.type == tp.type and not tp.wpt_id == last_edited.wpt_id:
            '''SSS changed'''
            tp.type = 'waypoint'
            edited = True
        elif ((tp.type == 'endspeed' and last_edited.type == tp.type and not tp.wpt_id == last_edited.wpt_id)
              or (any(t.type == 'speed' for t in tps)
                  and task.turnpoints.index(tp) < tps.index(next(t for t in tps if t.type == 'speed')))):
            '''ESS changed or SSS is after this tp'''
            tp.type = 'waypoint'
            edited = True
        elif tp.type == 'goal' and tps.index(tp) < tps.index(tps[-1]):
            tp.type = 'waypoint'
            tp.shape = 'circle'
            edited = True
    if edited:
        task.update_waypoints()
    if task.opt_dist or tps[-1].type == 'goal':
        task.calculate_optimised_task_length()
        task.calculate_task_length()
        task.update_task_distance()
        write_map_json(task_id)
    return get_task_turnpoints(task)


def get_task_turnpoints(task) -> dict:
    from airspaceUtils import read_airspace_map_file
    from task import get_map_json

    turnpoints = task.read_turnpoints()
    max_n = 0
    total_dist = ''
    for tp in turnpoints:
        tp['original_type'] = tp['type']
        tp['partial_distance'] = '' if not tp['partial_distance'] else c_round(tp['partial_distance'] / 1000, 2)
        if int(tp['num']) > max_n:
            max_n = int(tp['num'])
            total_dist = tp['partial_distance']
        if tp['type'] == 'speed':
            ''' using NO WPT DIRECTION for start as for other waypoints - FAI GAP RULES 2020 '''
            # if tp['how'] == 'entry':
            #     tp['type_text'] = 'SSS - Out/Enter'
            # else:
            #     tp['type_text'] = 'SSS - In/Exit'
            tp['type_text'] = 'SSS'
        elif tp['type'] == 'endspeed':
            tp['type_text'] = 'ESS'
        elif tp['type'] == 'goal':
            if tp['shape'] == 'circle':
                tp['type_text'] = 'Goal Cylinder'
            else:
                tp['type_text'] = 'Goal Line'
        else:
            tp['type_text'] = tp['type'].capitalize()
    if task.opt_dist is None or total_dist == '':
        total_dist = 'Distance not yet calculated'
    else:
        total_dist = str(total_dist) + "km"
    # max_n = int(math.ceil(max_n / 10.0)) * 10
    max_n += 1

    if task.opt_dist:
        '''task map'''
        task_coords, task_turnpoints, short_route, goal_line, \
            tol, min_tol, bbox, offset, airspace = get_map_json(task.id)
        layer = {'geojson': None, 'bbox': bbox}
        '''airspace'''
        show_airspace = False
        if airspace:
            airspace_layer = read_airspace_map_file(airspace)['spaces']
        else:
            airspace_layer = None

        task_map = make_map(
            layer_geojson=layer,
            points=task_coords,
            circles=task_turnpoints,
            polyline=short_route,
            goal_line=goal_line,
            margin=tol,
            min_margin=min_tol,
            waypoint_layer=True,
            airspace_layer=airspace_layer,
            show_airspace=show_airspace,
        )
        task_map = task_map._repr_html_()
    else:
        task_map = None

    return {
        'turnpoints': turnpoints,
        'next_number': max_n,
        'distance': total_dist,
        'map': task_map,
        'task_set': task.is_set,
    }


def check_task(task_id: int) -> tuple:
    """check all conditions for task"""
    from calcUtils import epoch_to_datetime
    from db.tables import TblResultFile as RF
    from db.tables import TblTaskResult as R
    from db.tables import TblForComp as F
    from db.tables import TblTask as T

    need_full_rescore = False
    need_new_scoring = False
    need_older_tracks_recheck = False

    with db_session() as db:
        '''get last track creation'''
        query = db.query(R.last_update).filter_by(task_id=task_id).filter(R.track_file.isnot(None))
        if query.count() > 0:
            last_track = query.order_by(R.last_update.desc()).first()
            first_track = query.order_by(R.last_update).first()
            last_file = db.query(RF).filter_by(task_id=task_id).order_by(RF.created.desc()).first()
            task = db.query(T).get(task_id)
            comp_id = task.comp_id
            formula_updated = db.query(F.last_update).filter_by(comp_id=comp_id).scalar()
            task_updated = max(formula_updated, task.last_update)
            last_file_created = None if not last_file else epoch_to_datetime(last_file.created)
            if last_file_created and last_file_created < max(last_track.last_update, task_updated):
                '''last results were calculated before last track or last formula changing'''
                need_new_scoring = True
            if task_updated > first_track.last_update:
                '''formula or task has changed after first track was evaluated'''
                need_older_tracks_recheck = True
            # todo logic to see if we need a full rescore, probably only if task was canceled and we have more tracks,
            #  or stopped and elapsed time / multistart with newer tracks started later than previous last
    return need_new_scoring, need_older_tracks_recheck, need_full_rescore


def get_outdated_tracks(task_id: int) -> list:
    from db.tables import TblTaskResult as R
    from db.tables import TblForComp as F
    from db.tables import TblTask as T
    with db_session() as db:
        task = db.query(T).get(task_id)
        comp_id = task.comp_id
        formula_updated = db.query(F.last_update).filter_by(comp_id=comp_id).scalar()
        task_updated = max(formula_updated, task.last_update)
        query = db.query(R.par_id).filter_by(task_id=task_id).filter(R.track_file.isnot(None))
        return [row.par_id for row in query.filter(R.last_update < task_updated).all()]


def get_comp_regions(compid: int):
    """Gets a list of dicts of: if defines.yaml waypoint library function is on - all regions
    otherwise only the regions with their comp_id field set the the compid parameter"""
    import Defines
    import region

    if Defines.WAYPOINT_AIRSPACE_FILE_LIBRARY:
        return region.get_all_regions()
    else:
        return region.get_comp_regions_and_wpts(compid)


def get_regions_used_in_comp(compid: int, tasks: bool = False) -> list:
    """returns a list of reg_id of regions used in a competition.
    Used for waypoints and area map link in competition page"""
    from db.tables import TblRegion as R
    from db.tables import TblTask as T

    regions = [el.reg_id for el in R.get_all(comp_id=compid)]
    if tasks:
        regions.extend([el.reg_id for el in T.get_all(comp_id=compid)])
        regions = list(set(regions))
    return [el for el in regions if el is not None]


def get_region_choices(compid: int):
    """gets a list of regions to be used in frontend select field (choices) and details of each region (details)"""
    regions = get_comp_regions(compid)
    choices = []
    details = {}
    for region in regions['regions']:
        choices.append((region['reg_id'], region['name']))
        details[region['reg_id']] = region
    return choices, details


def get_waypoint_choices(reg_id: int):
    import region

    wpts = region.get_region_wpts(reg_id)
    choices = [(wpt['rwp_id'], wpt['name'] + ' - ' + wpt['description']) for wpt in wpts]

    return choices, wpts


def get_pilot_list_for_track_management(taskid: int, recheck: bool) -> list:
    from pilot.flightresult import get_task_results

    pilots = get_task_results(taskid)
    outdated = [] if not recheck else get_outdated_tracks(taskid)
    all_data = []
    for pilot in pilots:
        data = {e: getattr(pilot, e) for e in ('ID', 'name')}
        data.update(track_result_output(pilot, task_id=taskid))
        data['outdated'] = data['par_id'] in outdated
        all_data.append(data)

    return all_data


def get_pilot_list_for_tracks_status(taskid: int):
    from db.tables import TblTaskResult as R

    pilots = [row._asdict() for row in R.get_task_results(taskid)]

    all_data = []
    for pilot in pilots:
        data = {
            'par_id': pilot['par_id'],
            'ID': pilot['ID'],
            'name': pilot['name'],
            'sex': pilot['sex'],
            'track_id': pilot['track_id'],
            'comment': pilot['comment'],
        }
        if pilot['track_file']:
            parid = data['par_id']
            if pilot['ESS_time']:
                time = sec_to_time(pilot['ESS_time'] - pilot['SSS_time'])
                if pilot['result_type'] == 'goal':
                    result = f'Goal {time}'
                else:
                    result = f"ESS {round(pilot['distance_flown'] / 1000, 2)} Km (<del>{time}</del>)"
            else:
                result = f"LO {round(pilot['distance_flown'] / 1000, 2)} Km"
            data['Result'] = f'<a href="/map/{parid}-{taskid}?back_link=0&full=1" target="_blank">{result}</a>'
        elif pilot['result_type'] == "mindist":
            data['Result'] = "Min Dist"
        else:
            data['Result'] = "Not Yet Processed" if not pilot['track_id'] else pilot['result_type'].upper()
        all_data.append(data)
    return all_data


def get_waypoint(wpt_id: int = None, rwp_id: int = None):
    """reads waypoint from tblTaskWaypoint or tblRegionWaypoint depending on input and returns Turnpoint object"""
    if not (wpt_id or rwp_id):
        return None
    with db_session() as db:
        if wpt_id:
            result = db.query(TblTaskWaypoint).get(wpt_id)
        else:
            result = db.query(TblRegionWaypoint).get(rwp_id)
        tp = Turnpoint()
        result.populate(tp)
    return tp


def save_turnpoint(task_id: int, turnpoint: Turnpoint):
    """save turnpoint in a task- for frontend"""
    if not (type(task_id) is int and task_id > 0):
        print("task not present in database ", task_id)
        return None
    with db_session() as db:
        if not turnpoint.wpt_id:
            '''add new taskWaypoint'''
            # tp = TblTaskWaypoint(**turnpoint.as_dict())
            tp = TblTaskWaypoint.from_obj(turnpoint)
            db.add(tp)
            db.flush()
        else:
            '''update taskWaypoint'''
            tp = db.query(TblTaskWaypoint).get(turnpoint.wpt_id)
            if tp:
                for k, v in turnpoint.as_dict().items():
                    if hasattr(tp, k):
                        setattr(tp, k, v)
            db.flush()
    return tp.wpt_id


def copy_turnpoints_from_task(task_id: int, task_from: int) -> bool:
    """Copy Task Turnpoints from another one"""
    from db.tables import TblTaskWaypoint as W
    objects = []
    with db_session() as db:
        origin = db.query(W.num, W.name, W.rwp_id, W.lat, W.lon,
                          W.altitude, W.description, W.time, W.type, W.how,
                          W.shape, W.angle, W.radius).filter_by(task_id=task_from).order_by(W.num).all()
        for row in origin:
            new = W(task_id=task_id, **row._asdict())
            objects.append(new)

        db.bulk_save_objects(objects=objects)
    return True


def allowed_tracklog(filename, extension=track_formats):
    ext = Path(filename).suffix
    if not ext:
        return False
    # Check if the extension is allowed (make everything uppercase)
    if ext.strip('.').lower() in [e.lower() for e in extension]:
        return True
    else:
        return False


def allowed_tracklog_filesize(filesize, size=5):
    """check if tracklog exceeds maximum file size for tracklog (5mb)"""
    if int(filesize) <= size * 1024 * 1024:
        return True
    else:
        return False


def process_igc(task_id: int, par_id: int, tracklog, user, check_g_record=False, check_validity=False):
    from airspace import AirspaceCheck
    from pilot.flightresult import FlightResult, save_track
    from trackUtils import check_flight, igc_parsing_config_from_yaml, import_igc_file, save_igc_file
    from task import Task
    from tempfile import mkdtemp
    from Defines import TEMPFILES

    if production():
        tmpdir = mkdtemp(dir=TEMPFILES)
        file = Path(tmpdir, tracklog.filename)
        tracklog.save(file)
        job = current_app.task_queue.enqueue(process_igc_background,
                                             task_id, par_id, file, user, check_g_record, check_validity)
        return True, None

    pilot = FlightResult.read(par_id, task_id)
    if not pilot.name:
        return False, 'Pilot settings are not correct, or not found.'

    task = Task.read(task_id)

    """import track"""
    if check_validity:
        FlightParsingConfig = igc_parsing_config_from_yaml(task.igc_config_file)
    else:
        FlightParsingConfig = igc_parsing_config_from_yaml('_overide')

    '''check igc file is correct'''
    mytrack, error = import_igc_file(tracklog.filename, task, FlightParsingConfig, check_g_record=check_g_record)
    if error:
        return False, error['text']

    pilot.track_file = save_igc_file(tracklog, task.file_path, task.date, pilot.name, pilot.ID)

    airspace = None if not task.airspace_check else AirspaceCheck.from_task(task)
    check_flight(pilot, mytrack, task, airspace, print=print)
    '''save to database'''
    save_track(pilot, task.id)

    print(f"track verified with task {task.task_id}\n")

    data = track_result_output(pilot, task_id)
    return data, None


def process_igc_background(task_id: int, par_id: int, file, user, check_g_record=False, check_validity=False):
    from trackUtils import import_igc_file, save_igc_file, igc_parsing_config_from_yaml, check_flight
    import json
    from airspace import AirspaceCheck
    from pilot.flightresult import FlightResult, save_track
    from task import Task

    print = partial(print_to_sse, id=par_id, channel=user)
    print('|open_modal')

    pilot = FlightResult.read(par_id, task_id)
    if not pilot.name:
        return False, 'Pilot settings are not correct, or not found.'

    task = Task.read(task_id)

    if check_validity:
        FlightParsingConfig = igc_parsing_config_from_yaml(task.igc_config_file)
    else:
        FlightParsingConfig = igc_parsing_config_from_yaml('_overide')

    data = {'par_id': pilot.par_id, 'track_id': pilot.track_id}
    '''check igc file is correct'''
    mytrack, error = import_igc_file(file, task, FlightParsingConfig, check_g_record=check_g_record)
    if error:
        '''error importing igc file'''
        print(f"Error: {error['text']}")
        data['text'] = error['text']
        print(f"{json.dumps(data)}|{error['code']}")
        return None

    pilot.track_file = save_igc_file(file, task.file_path, task.date, pilot.name, pilot.ID)
    print(f'IGC file saved: {pilot.track_file}')
    airspace = None if not task.airspace_check else AirspaceCheck.from_task(task)
    print('***************START*******************')
    check_flight(pilot, mytrack, task, airspace, print=print)
    '''save to database'''
    save_track(pilot, task.id)

    data = track_result_output(pilot, task.task_id)

    print(json.dumps(data) + '|result')
    print('***************END****************')

    return True


def track_result_output(pilot, task_id) -> dict:
    data = {'par_id': pilot.par_id, 'ID': pilot.ID, 'track_id': pilot.track_id, 'Result': '', 'notifications': ''}

    if not pilot.track_file:
        data['Result'] = ("Min Dist" if pilot.result_type == "mindist"
                          else "Not Yet Processed" if pilot.result_type == "nyp"
                          else pilot.result_type.upper())
    else:
        time = ''
        if pilot.ESS_time:
            time = sec_to_time(pilot.ss_time)
        if pilot.result_type == 'goal':
            data['Result'] = f'GOAL {time}'
        elif pilot.result_type == 'lo' and time:
            data['Result'] = f'ESS <del>{time}</del> ({c_round(pilot.distance / 1000, 2)} Km)'
        elif pilot.result_type == 'lo':
            data['Result'] = f"LO {c_round(pilot.distance / 1000, 2)} Km"
        if pilot.track_id:  # if there is a track, make the result a link to the map
            result = data['Result']
            data['Result'] = f'<a href="/map/{pilot.par_id}-{task_id}?back_link=0" target="_blank">{result}</a>'
        if pilot.notifications:
            data['notifications'] = f"{'<br />'.join(n.comment for n in pilot.notifications)}"
            data['Result'] += f'<a tabindex="0" class="p-1 ml-2" role="button" data-toggle="popover" ' \
                              f'data-container="body" data-trigger="focus" data-html="true" data-placement="top" ' \
                              f'title="Warning" data-content="{data["notifications"]}">' \
                              f'<span class="fas fa-exclamation-circle text-warning"></span></a>'

    return data


def unzip_igc(zipfile):
    """split function for background in production"""
    from os import chmod
    from tempfile import mkdtemp

    from Defines import TEMPFILES
    from trackUtils import extract_tracks

    """create a temporary directory"""
    # with TemporaryDirectory() as tracksdir:
    tracksdir = mkdtemp(dir=TEMPFILES)
    # make readable and writable by other users as background runs in another container
    chmod(tracksdir, 0o775)

    error = extract_tracks(zipfile, tracksdir)
    if error:
        print(f"An error occurred while dealing with file {zipfile}")
        return None
    return tracksdir


def process_archive_background(taskid: int, tracksdir, user, check_g_record=False, track_source=None):
    """function split for background use.
    tracksdir is a temp dir that will be deleted at the end of the function"""
    from shutil import rmtree

    from task import Task
    from trackUtils import assign_and_import_tracks, get_tracks

    print = partial(print_to_sse, id=None, channel=user)
    print('|open_modal')
    task = Task.read(taskid)
    if task.opt_dist == 0:
        print('task not optimised.. optimising')
        task.calculate_optimised_task_length()
    tracks = get_tracks(tracksdir)
    """find valid tracks"""
    if tracks is None:
        print(f"There are no valid tracks in zipfile")
        return None
    """associate tracks to pilots and import"""
    assign_and_import_tracks(tracks, task, track_source, user=user, check_g_record=check_g_record, print=print)
    rmtree(tracksdir)
    print('|reload')
    return 'Success'


def process_archive(task, zipfile, check_g_record=False, track_source=None):
    from tempfile import TemporaryDirectory

    from trackUtils import assign_and_import_tracks, extract_tracks, get_tracks

    if task.opt_dist == 0:
        print('task not optimised.. optimising')
        task.calculate_optimised_task_length()

    """create a temporary directory"""
    with TemporaryDirectory() as tracksdir:
        error = extract_tracks(zipfile, tracksdir)
        if error:
            print(f"An error occurred while dealing with file {zipfile} \n")
            return None
        """find valid tracks"""
        tracks = get_tracks(tracksdir)
        if not tracks:
            print(f"There are no valid tracks in zipfile {zipfile}, or all pilots are already been scored \n")
            return None

        """associate tracks to pilots and import"""
        assign_and_import_tracks(tracks, task, track_source, check_g_record=check_g_record)
        return 'Success'


def process_zip_file(zip_file: Path, taskid: int, username: str, grecord: bool, track_source: str = None):
    from task import Task

    if production():
        tracksdir = unzip_igc(zip_file)
        job = current_app.task_queue.enqueue(
            process_archive_background,
            taskid=taskid,
            tracksdir=tracksdir,
            user=username,
            check_g_record=grecord,
            track_source=track_source,
            job_timeout=2000,
        )
        resp = jsonify(success=True)
        return resp
    else:
        task = Task.read(taskid)
        data = process_archive(task, zip_file, check_g_record=grecord, track_source=track_source)
        resp = jsonify(success=True) if data == 'Success' else None
        return resp


def get_task_result_file_list(taskid: int, comp_id: int) -> dict:
    from db.tables import TblResultFile as R

    files = []
    with db_session() as db:
        task_results = db.query(R.created, R.filename, R.status, R.active, R.ref_id).filter_by(task_id=taskid).all()
        comp_results = (
            db.query(R.created, R.filename, R.status, R.active, R.ref_id)
            .filter(R.comp_id == comp_id, R.task_id.is_(None))
            .all()
        )

        tasks_files = [row._asdict() for row in task_results]
        comp_files = [row._asdict() for row in comp_results]
        return {'task': tasks_files, 'comp': comp_files}


def number_of_tracks_processed(taskid: int):
    from db.tables import TblParticipant as P
    from db.tables import TblTask as T
    from db.tables import TblTaskResult as R
    from sqlalchemy import func

    with db_session() as db:
        results = db.query(func.count()).filter(R.task_id == taskid).scalar()
        pilots = (
            db.query(func.count(P.par_id)).outerjoin(T, P.comp_id == T.comp_id).filter(T.task_id == taskid).scalar()
        )
    return results, pilots


def get_score_header(files, offset):
    import time

    from Defines import RESULTDIR

    active = None
    header = f"It has not been scored"
    file = next((el for el in files if int(el['active']) == 1), None)
    if file:
        active = file['filename']
        published = time.ctime(file['created'] + offset)
        '''check file exists'''
        if Path(RESULTDIR, file['filename']).is_file():
            header = 'Auto Generated Result ' if 'Overview' in file['filename'] else "Published Result "
            header += f"ran at: {published} "
            header += f"Status: {file['status'] if not file['status'] in ('', None, 'None') else 'No status'}"
        else:
            header = f"WARNING: Active result file is not found! (ran: {published})"
    elif len(files) > 0:
        header = "No published results"
    return header, active


def get_comp_users_ids(comp_id: int) -> list:
    """returns a list of ids for scorekeepers (and owner) of a competition"""
    from airscore.user.models import User
    from db.tables import TblCompAuth as CA

    all_ids = []
    with db_session() as db:
        result = (
            db.query(User.id)
            .join(CA, User.id == CA.user_id)
            .filter(CA.comp_id == comp_id, CA.user_auth.in_(('owner', 'admin')))
            .all()
        )
        if result:
            all_ids = [row.id for row in result]
        return all_ids


def get_comp_scorekeepers(compid_or_taskid: int, task_id=False) -> tuple:
    """returns owner and list of scorekeepers takes compid by default or taskid if taskid is True"""
    from db.tables import TblCompAuth as CA
    from airscore.user.models import User

    owner = None
    comp_scorekeepers = []
    available_scorekeepers = []

    with db_session() as db:
        '''comp scorekeepers'''
        if task_id:
            taskid = compid_or_taskid
            q1 = (
                db.query(User.id, User.username, User.first_name, User.last_name, CA.user_auth)
                .join(CA, User.id == CA.user_id)
                .join(TblTask, CA.comp_id == TblTask.comp_id)
                .filter(TblTask.task_id == taskid, CA.user_auth.in_(('owner', 'admin')))
                .all()
            )
        else:
            compid = compid_or_taskid
            q1 = (
                db.query(User.id, User.username, User.first_name, User.last_name, CA.user_auth)
                .join(CA, User.id == CA.user_id)
                .filter(CA.comp_id == compid, CA.user_auth.in_(('owner', 'admin')))
                .all()
            )

        '''available scorekeepers'''
        q2 = (
                db.query(User.id, User.first_name, User.last_name, User.username, User.access)
                .filter(User.id.notin_([a.id for a in q1]),
                        User.access.in_(['scorekeeper']),
                        User.active == 1)
                .all()
        )

        if q1:
            comp_scorekeepers = [row._asdict() for row in q1]
            '''owner'''
            owner = next((p for p in comp_scorekeepers if p['user_auth'] == 'owner'), None)
            if owner:
                comp_scorekeepers.remove(owner)
        if q2:
            available_scorekeepers = [row._asdict() for row in q2]

    return owner, comp_scorekeepers, available_scorekeepers


def delete_comp_scorekeeper(comp_id: int, user_id: int) -> bool:
    from db.tables import TblCompAuth as C
    try:
        C.get_one(comp_id=comp_id, user_id=user_id).delete()
        return True
    except Exception as e:
        # raise
        return False


def check_comp_editor(compid: int, user) -> bool:
    """ check if user is a scorer for the event"""
    if user.is_admin or user.is_manager:
        return True
    else:
        scorekeeper_ids = get_comp_users_ids(compid)
        return user.id in scorekeeper_ids


def change_comp_category(comp_id: int, new: str, formula_name: str) -> bool:
    from db.tables import TblCompRanking, TblForComp
    from formula import list_formulas
    try:
        with db_session() as db:
            comp = db.query(TblCompetition).get(comp_id)
            comp.comp_class = new
            db.query(TblCompRanking).filter_by(comp_id=comp_id, rank_type='cert').delete(synchronize_session=False)
            db.flush()
            formulas = list_formulas().get(new)
            if formula_name not in formulas:
                formula_name = formulas[0]
            formula, preset = get_comp_formula_preset(comp_id, formula_name, new)
            row = db.query(TblForComp).get(comp_id)
            row.from_obj(formula)
            db.flush()
        return True
    except:
        raise
        return False


def get_comp_formula_preset(comp_id: int, formula_name: str, comp_class: str) -> tuple:
    from formula import Formula
    formula = Formula.read(comp_id)
    formula.reset(comp_class, formula_name)
    preset = formula.get_preset()
    return formula, preset


def set_comp_scorekeeper(compid: int, userid, owner=False):
    from db.tables import TblCompAuth as CA

    auth = 'owner' if owner else 'admin'
    with db_session() as db:
        admin = CA(user_id=userid, comp_id=compid, user_auth=auth)
        db.add(admin)
        db.flush()
    return True


def get_all_users():
    """returns a list of all scorekeepers in the system"""
    from airscore.user.models import User

    with db_session() as db:
        all_users = db.query(
            User.id, User.username, User.first_name, User.last_name, User.access, User.email, User.active, User.nat
        ).all()
        if all_users:
            all_users = [row._asdict() for row in all_users]
        return all_users


def generate_random_password() -> str:
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for i in range(20))  # for a 20-character password
    return password


def generate_serializer():
    from itsdangerous import URLSafeTimedSerializer
    from airscore.settings import SECRET_KEY
    return URLSafeTimedSerializer(SECRET_KEY)


def generate_confirmation_token(email):
    from airscore.settings import SECURITY_PASSWORD_SALT
    serializer = generate_serializer()
    return serializer.dumps(email, salt=SECURITY_PASSWORD_SALT)


def confirm_token(token, expiration=86400):
    from airscore.settings import SECURITY_PASSWORD_SALT
    serializer = generate_serializer()
    try:
        email = serializer.loads(
            token,
            salt=SECURITY_PASSWORD_SALT,
            max_age=expiration
        )
    except:
        return False
    return email


def send_email(recipients, subject, text_body, html_body, sender=None):
    from airscore.extensions import mail
    from airscore.settings import ADMINS
    try:
        mail.send_message(
            recipients=recipients,
            subject=subject,
            body=text_body,
            html=html_body,
            sender=sender or ADMINS
        )
        return True, None
    except:
        # raise
        return False, f"Error trying to send mail."


def update_airspace_file(old_filename, new_filename):
    """change the name of the openair file in all regions it is used."""
    R = aliased(TblRegion)
    with db_session() as db:
        db.query(R).filter(R.openair_file == old_filename).update(
            {R.openair_file: new_filename}, synchronize_session=False
        )
        db.commit()
    return True


# def save_waypoint_file(file):
#     from Defines import WAYPOINTDIR, AIRSPACEDIR
#     full_file_name = path.join(WAYPOINTDIR, filename)


def get_non_registered_pilots(compid: int):
    from db.tables import PilotView, TblParticipant

    p = aliased(TblParticipant)
    pv = aliased(PilotView)

    with db_session() as db:
        '''get registered pilots'''
        reg = db.query(p.pil_id).filter(p.comp_id == compid).subquery()
        non_reg = (
            db.query(pv.pil_id, pv.civl_id, pv.first_name, pv.last_name)
            .filter(reg.c.pil_id == None)
            .outerjoin(reg, reg.c.pil_id == pv.pil_id)
            .order_by(pv.first_name, pv.last_name)
            .all()
        )

        non_registered = [row._asdict() for row in non_reg]
    return non_registered


def get_igc_parsing_config_file_list():
    yaml = ruamel.yaml.YAML()
    configs = []
    choices = []
    for file in scandir(IGCPARSINGCONFIG):
        if file.name.endswith(".yaml") and not file.name.startswith("_"):
            with open(file.path) as fp:
                config = yaml.load(fp)
            configs.append(
                {
                    'file': file.name,
                    'name': file.name[:-5],
                    'description': config['description'],
                    'editable': config['editable'],
                }
            )
            choices.append((file.name[:-5], file.name[:-5]))
    return choices, configs


def get_comps_with_igc_parsing(igc_config):
    from db.tables import TblCompetition

    c = aliased(TblCompetition)
    with db_session() as db:
        return db.query(c.comp_id).filter(c.igc_config_file == igc_config).all()


def get_comp_info(compid: int, task_ids=None):
    if task_ids is None:
        task_ids = []
    c = aliased(TblCompetition)
    t = aliased(TblTask)

    with db_session() as db:
        non_scored_tasks = (
            db.query(t.task_id.label('id'),
                     t.task_name, t.date, t.task_type, t.opt_dist, t.training, t.comment, t.cancelled)
            .filter_by(comp_id=compid)
            .order_by(t.date.desc())
            .all()
        )

        competition_info = (
            db.query(c.comp_id, c.comp_name, c.comp_site, c.date_from, c.date_to, c.self_register, c.website)
            .filter(c.comp_id == compid)
            .one()
        )
        comp = competition_info._asdict()

        return comp, [row._asdict() for row in non_scored_tasks if row.id not in task_ids]


def get_participants(compid: int, source='all'):
    """get all registered pilots for a comp.
    Compid: comp_id
    source: all: all participants
            internal: only participants from pilot table (with pil_id)
            external: only participants not in pilot table (without pil_id)"""
    from compUtils import get_participants
    from formula import Formula

    pilots = get_participants(compid)
    pilot_list = []
    external = 0
    for pilot in pilots:
        if pilot.nat_team == 1:
            pilot.nat_team = '✓'
        else:
            pilot.nat_team = None
        if pilot.paid == 1:
            pilot.paid = 'Y'
        else:
            pilot.paid = 'N'
        if source == 'all' or source == 'internal':
            if pilot.pil_id:
                pilot_list.append(pilot.as_dict())
        if source == 'all' or source == 'external':
            if not pilot.pil_id:
                external += 1
                pilot_list.append(pilot.as_dict())
    formula = Formula.read(compid)
    teams = {
        'country_scoring': formula.country_scoring,
        'max_country_size': formula.max_country_size,
        'country_size': formula.country_size,
        'team_scoring': formula.team_scoring,
        'team_size': formula.team_size,
        'max_team_size': formula.max_team_size,
    }
    return pilot_list, external, teams


def check_team_size(compid: int, nat=False):
    """Checks that the number of pilots in a team don't exceed the allowed number"""
    from db.tables import TblParticipant as P
    from formula import Formula

    formula = Formula.read(compid)
    message = ''
    if nat:
        max_team_size = formula.max_country_size or 0
    else:
        max_team_size = formula.max_team_size or 0

    with db_session() as db:
        if nat:
            q = db.query(P.nat, func.sum(P.nat_team)).filter(P.comp_id == compid).group_by(P.nat)
        else:
            q = db.query(P.team, func.count(P.team)).filter(P.comp_id == compid).group_by(P.team)
        result = q.all()
        for team in result:
            if team[1] > max_team_size:
                message += f"<p>Team {team[0]} has {team[1]} members - only {max_team_size} allowed.</p>"
    return message


def print_to_sse(text, id, channel):
    """Background jobs can send SSE by using this function which takes a string and sends to webserver
    as an HTML post request (via push_sse).
    A message type can be specified by including it in the string after a pipe "|" otherwise the default message
    type is 'info'
    Args:
        :param text: a string
        :param id: int/string to identify what the message relates to (par_id etc.)
        :param channel: string to identify destination of message (not access control) such as username etc
    """
    message = text.split('|')[0]
    if len(text.split('|')) > 1:
        message_type = text.split('|')[1]
    else:
        message_type = 'info'
    body = {'message': message, 'id': id}
    push_sse(body, message_type, channel=channel)


def push_sse(body, message_type, channel):
    """send a post request to webserver with contents of SSE to be sent"""
    data = {'body': body, 'type': message_type, 'channel': channel}
    requests.post(
        f"http://{environ.get('FLASK_CONTAINER')}:" f"{environ.get('FLASK_PORT')}/internal/see_message", json=data
    )


def production():
    """Checks if we are running production or dev via environment variable."""
    return not environ['FLASK_DEBUG'] == '1'


def unique_filename(filename, filepath):
    """checks file does not already exist and creates a unique and secure filename"""
    import glob
    from os.path import join
    from pathlib import Path

    from werkzeug.utils import secure_filename

    fullpath = join(filepath, filename)
    if Path(fullpath).is_file():
        index = str(len(glob.glob(fullpath)) + 1).zfill(2)
        name, suffix = filename.rsplit(".", 1)
        filename = '_'.join([name, index]) + '.' + suffix
    return secure_filename(filename)


def get_pretty_data(content: dict, export=False) -> dict or str:
    """transforms result json file in human readable data"""
    from result import get_startgates, pretty_format_results
    from calcUtils import get_date

    try:
        '''time offset'''
        timeoffset = 0 if 'time_offset' not in content['info'].keys() else int(content['info']['time_offset'])
        '''result file type'''
        result_type = content['file_stats']['result_type']
        '''score decimals'''
        td = (
            0
            if 'formula' not in content.keys() or 'task_result_decimal' not in content['formula'].keys()
            else int(content['formula']['task_result_decimal'])
        )
        cd = (
            0
            if 'formula' not in content.keys() or 'comp_result_decimal' not in content['formula'].keys()
            else int(content['formula']['comp_result_decimal'])
        )
        pretty_content = dict()
        if 'file_stats' in content.keys():
            pretty_content['file_stats'] = pretty_format_results(content['file_stats'], timeoffset)
        pretty_content['info'] = pretty_format_results(content['info'], timeoffset)
        if 'comps' in content.keys():
            pretty_content['comps'] = pretty_format_results(content['comps'], timeoffset)
        if 'tasks' in content.keys():
            pretty_content['tasks'] = pretty_format_results(content['tasks'], timeoffset)
        elif 'route' in content.keys():
            pretty_content['info'].update(startgates=get_startgates(content['info']))
            pretty_content['route'] = pretty_format_results(content['route'], timeoffset)
        if 'stats' in content.keys():
            pretty_content['stats'] = pretty_format_results(content['stats'], timeoffset)
        if 'formula' in content.keys():
            pretty_content['formula'] = pretty_format_results(content['formula'])
        if 'results' in content.keys():
            results = []
            '''rankings'''
            rankings = [dict(rank=1, counter=0, prev=None, **rank) for rank in content['rankings']]

            rank = 0
            prev = None
            for idx, r in enumerate(content['results'], 1):
                # p = pretty_format_results(r, timeoffset, td, cd)
                '''rankings'''
                if result_type == 'comp' or r['result_type']:
                    d = cd if result_type == 'comp' else td
                    r['score'] = c_round(r['score'] or 0, d)
                    if not prev == r['score']:
                        rank, prev = idx, r['score']
                    r['rank'] = rank
                    r['rankings'] = {}
                    for s in rankings:
                        if s['rank_type'] == 'overall':
                            r['rankings'][s['rank_id']] = rank
                            s['counter'] += 1
                        elif (
                                (s['rank_type'] == 'cert' and r['glider_cert'] in s['certs'])
                                or (s['rank_type'] == 'female' and r['sex'] == 'F')
                                or (s['rank_type'] == 'nat' and r['nat'] == s['nat'])
                                or (s['rank_type'] == 'custom' and 'custom' in r.keys()
                                    and str(s['attr_id']) in r['custom'].keys()
                                    and r['custom'][str(s['attr_id'])] == s['rank_value'])
                                or (s['rank_type'] == 'birthdate' and 'birthdate' in r.keys()
                                    and isinstance(get_date(r['birthdate']), datetime.date)
                                    and (
                                            (s['min_date'] and get_date(s['min_date']) <= get_date(r['birthdate']))
                                            or (s['max_date'] and get_date(s['max_date']) >= get_date(r['birthdate']))
                                    ))
                        ):
                            s['counter'] += 1
                            if not s['prev'] == r['score']:
                                s['rank'], s['prev'] = s['counter'], r['score']
                            r['rankings'][s['rank_id']] = f"{s['rank']} ({r['rank']})"
                        else:
                            r['rankings'][s['rank_id']] = ''
                    if result_type == 'comp':
                        r['name'] = f"<span class='sex-{r['sex']}'><b>{r['name']}</b></span>"
                        '''task results format'''
                        for k, v in r['results'].items():
                            if v['score'] == v['pre']:
                                v['score'] = f"{v['score']:.{td}f}"
                            else:
                                v['score'] = f"{v['score']:.{td}f} <del>{v['pre']:.{td}f}</del>"
                        r['score'] = f"<strong>{r['score']:.{cd}f}</strong>"
                        r = pretty_format_results(r, timeoffset)
                    elif result_type == 'task':
                        task_id = content['info']['id']
                        stopped = content['info']['stopped_time']
                        goal_alt = content['info']['goal_altitude']
                        if export or not r['track_file']:
                            r['name'] = f"<span class='sex-{r['sex']}'><b>{r['name']}</b></span>"
                        else:
                            r['name'] = f"<a class='sex-{r['sex']}' href='/map/{r['par_id']}-{task_id}'>" \
                                        f"<b>{r['name']}</b></a>"
                        if r['penalty']:
                            p = r['penalty']
                            style = f"{'danger' if p > 0 else 'success'}"
                            r['penalty'] = f"<strong class='text-{style}'>{p:.{td}f}</strong>"
                            r['score'] = f"<strong class='text-{style}'>{r['score']:.{td}f}</strong>"
                        else:
                            r['score'] = f"<strong>{r['score']:.{td}f}</strong>"
                        r = pretty_format_results(r, timeoffset)
                        goal = r['goal_time']
                        r['ESS_time'] = r['ESS_time'] if goal else f"<del>{r['ESS_time']}</del>"
                        r['speed'] = r['speed'] if goal else f"<del>{r['speed']}</del>"
                        r['ss_time'] = r['ss_time'] if goal else f"<del>{r['ss_time']}</del>"
                        if stopped and r['stopped_altitude']:
                            r['stopped_altitude'] = f"+{max(0, r['stopped_altitude'] - goal_alt)}"
                        # ab = ''  # alt bonus
                results.append(r)
            pretty_content['results'] = results
            pretty_content['rankings'] = [
                {k: c[k] for k in ('rank_id', 'rank_name', 'description', 'counter')} for c in rankings
            ]

        return pretty_content
    except Exception:
        # raise
        return 'error'


def adjust_task_result(task_id: int, filename: str, par_id: int, notifications: list):
    from pilot.flightresult import FlightResult
    from pilot.notification import Notification
    from formula import get_formula_lib_by_name
    from result import open_json_file, order_task_results, write_json_file
    from task import Task
    data = open_json_file(filename)
    result = next((p for p in data['results'] if p['par_id'] == par_id), None)
    task = Task.create_from_json(task_id, filename)
    lib = get_formula_lib_by_name(data['formula']['formula_name'])
    if result and task and lib:
        try:
            '''create FlightResult obj'''
            pilot = FlightResult.from_result(result)
            pilot.notifications = [Notification.from_dict(d) for d in notifications]
            '''calculate penalty and score'''
            penalty, score = lib.pilot_penalty(task, pilot)
            '''check against day_quality, max score = day_quality * 1000'''
            result['penalty'] = penalty
            result['score'] = score
            result['notifications'] = [n.as_dict() for n in pilot.notifications]
            result['comment'] = pilot.comment
            data['results'] = order_task_results(data['results'])
            write_json_file(filename, data)
            return True
        except:
            # raise
            print(f'Error trying to update result file {filename}: par_id {par_id}')
    return False


def full_rescore(taskid: int, background=False, status=None, autopublish=None, compid=None, user=None):
    from task import Task

    task = Task.read(taskid)
    if background:
        print = partial(print_to_sse, id=None, channel=user)
        print('|open_modal')
        print('***************START*******************')
        refid, filename = task.create_results(mode='full', status=status, print=print)
        if refid and autopublish:
            publish_task_result(taskid, filename)
            if compid:
                update_comp_result(compid, name_suffix='Overview')
        print('****************END********************')
        print(f'{filename or "ERROR"}|reload_select_latest')
        return None
    else:
        refid, filename = task.create_results(mode='full', status=status)
        if refid and autopublish:
            publish_task_result(taskid, filename)
            if compid:
                update_comp_result(compid, name_suffix='Overview')
        return refid


def get_task_igc_zip(task_id: int):
    import shutil

    from Defines import track_formats
    from trackUtils import get_task_fullpath

    task_path = get_task_fullpath(task_id)
    task_folder = task_path.parts[-1]
    comp_folder = task_path.parent
    zip_filename = task_folder + '.zip'
    zip_full_filename = Path(comp_folder, zip_filename)
    # check if there is a zip already there and is the youngest file for the task,
    # if not delete (if there) and (re)create
    if zip_full_filename.is_file():
        zip_time = zip_full_filename.stat().st_mtime
        list_of_files = [e for e in task_path.iterdir() if e.is_file() and e.suffix.strip('.').lower() in track_formats]
        latest_file = max(file.stat().st_mtime for file in list_of_files)
        if latest_file > zip_time:
            zip_full_filename.unlink(missing_ok=True)
        else:
            return zip_full_filename
    shutil.make_archive(comp_folder / task_folder, 'zip', task_path)
    return zip_full_filename


def check_short_code(comp_short_code):
    with db_session() as db:
        code = db.query(TblCompetition.comp_code).filter(TblCompetition.comp_code == comp_short_code).first()
        if code:
            return False
        else:
            return True


def import_participants_from_fsdb(comp_id: int, file: Path, from_CIVL=False) -> dict:
    """read the fsdb file"""
    from fsdb import FSDB
    from pilot.participant import unregister_all
    from ranking import delete_meta

    try:
        fsdb = FSDB.read(file, from_CIVL=from_CIVL)
        if len(fsdb.comp.participants) == 0:
            return dict(success=False, error='Error: not a valid FSDB file or has no participants.')

        if fsdb.custom_attributes:
            delete_meta(comp_id=comp_id)
            fsdb.add_custom_attributes(comp_id=comp_id)
        unregister_all(comp_id=comp_id)
        if fsdb.add_participants(comp_id=comp_id):
            return dict(success=True)
        return dict(success=False, error='Error: Participants were not imported correctly.')
    except (FileNotFoundError, TypeError, Exception):
        return dict(success=False, error='Internal error trying to parse FSDB file.')


def import_participants_from_excel_file(comp_id: int, excel_file: Path, comp_class: str = None) -> dict:
    from pilot.participant import unregister_all, extract_participants_from_excel, mass_import_participants
    from ranking import delete_meta
    from db.tables import TblCompetition as C

    if not comp_class:
        comp_class = C.get_by_id(comp_id).comp_class
    certs = [el['cert_name'] for el in get_certifications_details().get(comp_class)]
    pilots, custom_attributes = extract_participants_from_excel(comp_id, excel_file, certs)
    if not pilots:
        return jsonify(success=False, error='Error: not a valid excel file or has no participants.')
    if custom_attributes:
        delete_meta(comp_id)
        for attr in custom_attributes:
            attr.to_db()
        for pil in pilots:
            '''creates Participant.custom'''
            for attr in pil.attributes:
                attr_id = next((el.attr_id for el in custom_attributes if el.attr_value == attr['attr_value']), None)
                pil.custom[attr_id] = attr['meta_value']
            pil.attributes = None
    unregister_all(comp_id)
    if mass_import_participants(comp_id, pilots, check_ids=False):
        return dict(success=True)
    return dict(success=False, error='Error: Participants were not imported correctly.')


def comp_has_taskresults(comp_id: int) -> bool:
    """check if participants have already been scored in any task"""
    from db.tables import FlightResultView as F
    return len(F.get_all(comp_id=comp_id)) > 0


def create_participants_html(comp_id: int) -> (str, dict) or None:
    from comp import Comp

    try:
        comp = Comp.read(comp_id)
        return comp.create_participants_html()
    except Exception:
        return None


def create_participants_fsdb(comp_id: int) -> (str, str) or None:
    from fsdb import FSDB

    try:
        return FSDB.create_participants(comp_id)
    except Exception:
        return None


def create_task_html(file: str) -> (str, dict or list) or None:
    from result import TaskResult

    try:
        return TaskResult.to_html(file)
    except Exception:
        return None


def create_comp_html(comp_id: int) -> (str, dict or list) or None:
    from compUtils import get_comp_json_filename
    from result import CompResult

    try:
        file = get_comp_json_filename(comp_id)
        return CompResult.to_html(file)
    except Exception:
        return None


def create_inmemory_zipfile(files: list):
    import io
    import time
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        now = time.localtime(time.time())[:6]
        for el in files:
            name = el['filename']
            info = zipfile.ZipInfo(name)
            info.date_time = now
            info.compress_type = zipfile.ZIP_DEFLATED
            zip_file.writestr(info, el['content'])
    return zip_buffer


def render_html_file(content: dict) -> str:
    """render export html template:
    dict(title: str, headings: list, tables: list, timestamp: str)"""
    from flask import render_template

    return render_template(
        '/users/export_template.html',
        title=content['title'],
        headings=content['headings'],
        tables=content['tables'],
        timestamp=content['timestamp'],
    )


def create_stream_content(content):
    """ returns bytelike object"""
    from io import BytesIO

    mem = BytesIO()
    try:
        if isinstance(content, str):
            mem.write(content.encode('utf-8'))
        elif isinstance(content, BytesIO):
            mem.write(content.getvalue())
        else:
            mem.write(content)
        mem.seek(0)
        return mem
    except TypeError:
        return None


def list_countries() -> list:
    """Lists all countries with IOC code stored in database.
    :returns a list of dicts {name, code}"""
    from db.tables import TblCountryCode

    clist = TblCountryCode.get_list()
    return clist or []


def list_track_sources() -> list:
    """Lists all track sources enabled in Defines.
    :returns a list of (value, text)."""
    from Defines import track_sources

    sources = [('', ' -')]
    for el in track_sources:
        sources.append((el, el))
    return sources


def list_gmt_offset() -> list:
    """Lists GMT offsets.
    :returns a list of (value, text)."""
    tz = -12.0

    offsets = []
    while tz <= 14:
        offset = int(tz * 3600)
        sign = '-' if tz < 0 else '+'
        i, d = divmod(abs(tz), 1)
        h = int(i)
        m = '00' if not d else int(d * 60)
        text = f"{sign}{h}:{m}"
        offsets.append((offset, text))
        if tz in (5.5, 8.5, 12.5):
            odd = int((tz + 0.25) * 3600)
            offsets.append((odd, f"{sign}{h}:45"))
        tz += 0.5

    return offsets


def list_ladders(day: datetime.date = datetime.datetime.now().date(), ladder_class: str = None) -> list:
    """Lists all ladders stored in database, if ladders are active in settings.
    :returns a list."""
    from calcUtils import get_season_dates
    from Defines import LADDERS

    if not LADDERS:
        ''' Ladders are disabled in Settings'''
        return []
    ladders = []
    results = [el for el in get_ladders()]
    for el in results:
        '''create start and end dates'''
        starts, ends = get_season_dates(
            ladder_id=el['ladder_id'], season=int(el['season']), date_from=el['date_from'], date_to=el['date_to']
        )
        if starts < day < ends and (ladder_class is None or el['ladder_class'] == ladder_class):
            ladders.append(el)

    return ladders


def get_comp_ladders(comp_id: int) -> list:
    from db.tables import TblLadderComp as LC

    return [el.ladder_id for el in LC.get_all(comp_id=comp_id)]


def save_comp_ladders(comp_id: int, ladder_ids: list or None) -> bool:
    from db.tables import TblLadderComp as LC

    try:
        '''delete previous entries'''
        LC.delete_all(comp_id=comp_id)
        if ladder_ids:
            '''save entries'''
            results = []
            for el in ladder_ids:
                results.append(LC(comp_id=comp_id, ladder_id=el))
            LC.bulk_create(results)
        return True
    except Exception:
        # raise
        return False


def get_comp_result_files(comp_id: int, offset: int = None) -> dict:
    import time
    from Defines import RESULTDIR
    from db.tables import TblResultFile as R, TblCompetition as C
    from result import open_json_file

    with db_session() as db:
        query = (
            db.query(R.created, R.filename, R.status, R.active, R.ref_id)
            .filter(R.comp_id == comp_id, R.task_id.is_(None))
            .all()
        )
        comp_results = [row._asdict() for row in query]
        if not offset:
            offset = db.query(C).get(comp_id).time_offset

    comp_header, comp_active = get_score_header(comp_results, offset)
    comp_choices = []

    for file in comp_results:
        published = time.ctime(file['created'] + offset)
        if not Path(RESULTDIR, file['filename']).is_file():
            status = f"FILE NOT FOUND"
            tasks = []
        else:
            status = '' if file['status'] in [None, 'None'] else file['status']
            if 'Overview' in file['filename']:
                status = 'Auto Generated ' + status
            data = open_json_file(Path(RESULTDIR, file['filename']))
            tasks = data.get('tasks')
        comp_choices.append(dict(filename=file['filename'],
                                 text=f'{published} - {status}' if status else f'{published}',
                                 status=status,
                                 timestamp=published,
                                 tasks=tasks))
    comp_choices.reverse()

    return {
        'comp_choices': comp_choices,
        'comp_header': comp_header,
        'comp_active': comp_active,
    }


def get_task_result_files(task_id: int, comp_id: int = None, offset: int = None) -> dict:
    import time

    from compUtils import get_comp, get_comp_json, get_offset
    from Defines import RESULTDIR

    if not offset:
        offset = get_offset(task_id)
    files = get_task_result_file_list(task_id, comp_id or get_comp(task_id))

    task_header, task_active = get_score_header(files['task'], offset)
    comp_header, comp_active = get_score_header(files['comp'], offset)
    task_choices = []
    comp_choices = []

    for file in files['task']:
        published = time.ctime(file['created'] + offset)
        if not Path(RESULTDIR, file['filename']).is_file():
            status = f"FILE NOT FOUND"
        else:
            status = '' if file['status'] in [None, 'None'] else file['status']
        task_choices.append((file['filename'], f'{published} - {status}' if status else f'{published}'))
    task_choices.reverse()

    for file in files['comp']:
        published = time.ctime(file['created'] + offset)
        if not Path(RESULTDIR, file['filename']).is_file():
            status = f"FILE NOT FOUND"
        else:
            status = '' if file['status'] in [None, 'None'] else file['status']
            if 'Overview' in file['filename']:
                status = 'Auto Generated ' + status
        comp_choices.append((file['filename'], f'{published} - {status}' if status else f'{published}'))
    comp_choices.reverse()

    return {
        'task_choices': task_choices,
        'task_header': task_header,
        'task_active': task_active,
        'comp_choices': comp_choices,
        'comp_header': comp_header,
        'comp_active': comp_active,
    }


def get_comp_json_zip(comp_id: int):
    from db.tables import TblResultFile as R
    from Defines import RESULTDIR, TEMPFILES
    from zipfile import ZipFile

    results = R.get_all(comp_id=comp_id, active=1)
    zipfile = Path(TEMPFILES, f"json_files_{comp_id}.zip")
    if zipfile.is_file():
        zipfile.unlink(missing_ok=True)

    zipObj = ZipFile(zipfile, 'w')

    for f in results:
        file = Path(RESULTDIR, f.filename)
        if file.is_file():
            zipObj.write(file, file.name)

    zipObj.close()
    return zipfile


def get_region_waypoints(reg_id: int, region=None, openair_file: str = None) -> tuple:
    from mapUtils import create_airspace_layer, create_waypoints_layer
    from db.tables import TblRegion as R

    _, waypoints = get_waypoint_choices(reg_id)
    points_layer, bbox = create_waypoints_layer(reg_id)
    airspace_layer = None
    airspace_list = []
    if not openair_file:
        if region:
            openair_file = region.openair_file
        else:
            openair_file = R.get_by_id(reg_id).openair_file
    if openair_file:
        airspace_layer, airspace_list, _ = create_airspace_layer(openair_file)

    region_map = make_map(
        points=points_layer, circles=points_layer, airspace_layer=airspace_layer, show_airspace=False, bbox=bbox
    )
    return waypoints, region_map, airspace_list, openair_file


def get_task_airspace(task_id: int):
    from db.tables import TblTask, TblAirspaceCheck
    from task import get_map_json
    from mapUtils import create_airspace_layer
    task = TblTask.get_by_id(task_id)
    openair_file = task.openair_file

    if not openair_file:
        return None, None, None, None

    wpt_coords, turnpoints, short_route, goal_line, _, _, bbox, _, _ = get_map_json(task_id)
    airspace_layer, airspace_list, _ = create_airspace_layer(openair_file)
    airspace_map = make_map(
        points=wpt_coords, circles=turnpoints, polyline=short_route, airspace_layer=airspace_layer,
        goal_line=goal_line, show_airspace=True, bbox=bbox
    )
    query = TblAirspaceCheck.get_all(comp_id=task.comp_id)
    if any(el.task_id == task_id for el in query):
        result = next(el for el in query if el.task_id == task_id)
    else:
        result = next(el for el in query if el.task_id is None)
    parameters = {
        key: getattr(result, key) for key in (
            'v_boundary_penalty', 'v_outer_limit', 'h_inner_limit', 'h_boundary_penalty', 'h_outer_limit',
            'notification_distance', 'v_inner_limit', 'v_boundary', 'h_max_penalty', 'h_boundary', 'function',
            'v_max_penalty')
    }
    return openair_file, airspace_map, airspace_list, parameters


def unpublish_task_result(task_id: int):
    """Unpublish any active result"""
    from result import unpublish_result

    unpublish_result(task_id)


def publish_task_result(task_id: int, filename: str) -> bool:
    """Unpublish any active result, and publish the given one"""
    from result import publish_result, unpublish_result

    try:
        unpublish_result(task_id)
        publish_result(filename)
        return True
    except (FileNotFoundError, Exception) as e:
        print(f'Error trying to publish result')
        return False


def unpublish_comp_result(comp_id: int):
    """Unpublish any active result"""
    from result import unpublish_result

    unpublish_result(comp_id, comp=True)


def publish_comp_result(comp_id: int, filename: str) -> bool:
    """Unpublish any active result, and publish the given one"""
    from result import publish_result, unpublish_result

    try:
        unpublish_result(comp_id, comp=True)
        publish_result(filename)
        return True
    except (FileNotFoundError, Exception) as e:
        print(f'Error trying to publish result')
        return False


def publish_all_results(comp_id: int):
    """used for imported event autopublish:
    sets active all results of a given comp, assuming there is only one per task and final"""
    from db.tables import TblResultFile as R
    from db.conn import db_session
    from comp import Comp
    with db_session() as db:
        results = db.query(R).filter_by(comp_id=comp_id)
        for row in results:
            row.active = 1
    '''update comp result'''
    Comp.create_results(comp_id, status='Created from FSDB imported results', name_suffix='Overview')


def update_comp_result(comp_id: int, status: str = None, name_suffix: str = None) -> tuple:
    """Unpublish any active result, and creates a new one"""
    from comp import Comp

    try:
        _, ref_id, filename, timestamp = Comp.create_results(comp_id, status=status, name_suffix=name_suffix)
    except (FileNotFoundError, Exception) as e:
        print(f'Comp results creation error. Probably we miss some task results files?')
        return False, None, None
    return ref_id, filename, timestamp


def task_has_valid_results(task_id: int) -> bool:

    from db.tables import TblTaskResult as TR

    return bool(TR.get_task_flights(task_id))


def get_task_info(task_id: int) -> dict:

    from task import Task
    import time

    result = {}
    task = Task.read(task_id)
    if task:
        result = task.create_json_elements()
        result['file_stats'] = {'result_type': 'task', "timestamp": int(time.time()), "status": "tracks status"}
    return result


def convert_external_comp(comp_id: int) -> bool:
    from db.tables import TblCompetition as C
    from db.tables import TblNotification as N
    from db.tables import TblTask as T
    from db.tables import TblTaskResult as R
    from db.tables import TblTrackWaypoint as TW
    from sqlalchemy.exc import SQLAlchemyError
    from task import Task

    try:
        with db_session() as db:
            tasks = [el for el, in db.query(T.task_id).filter_by(comp_id=comp_id).distinct()]
            if tasks:
                '''clear tasks results'''
                results = (
                    db.query(R).filter(R.task_id.in_(tasks)).filter(R.result_type.notin_(['abs', 'dnf', 'mindist']))
                )
                if results:
                    tracks = [el.track_id for el in results.all()]
                    db.query(TW).filter(TW.track_id.in_(tracks)).delete(synchronize_session=False)
                    db.query(N).filter(N.track_id.in_(tracks)).delete(synchronize_session=False)
                    results.delete(synchronize_session=False)
                '''recalculate task distances'''
                for task_id in tasks:
                    task = Task.read(task_id)
                    '''get projection'''
                    task.create_projection()
                    task.calculate_task_length()
                    task.calculate_optimised_task_length()
                    '''Storing calculated values to database'''
                    task.update_task_distance()
            '''unset external flag'''
            comp = C.get_by_id(comp_id)
            comp.update(external=0)
            return True
    except (SQLAlchemyError, Exception):
        print(f'There was an Error trying to convert comp ID {comp_id}.')
        return False


def check_openair_file(file) -> tuple:
    from shutil import copyfile
    from tempfile import TemporaryDirectory

    from airspaceUtils import openair_content_to_data, save_airspace_map_check_files
    from Defines import AIRSPACEDIR

    modified = False
    with TemporaryDirectory() as tempdir:
        tempfile = Path(tempdir, file.filename)
        file.seek(0)
        file.save(tempfile)

        filename = None
        try:
            with open(tempfile, 'r+', encoding="utf-8") as fp:
                record_number, airspace_list, mapspaces, checkspaces, bbox = openair_content_to_data(fp)
        except UnicodeDecodeError:
            '''try different encoding'''
            with open(tempfile, 'r+', encoding="latin-1") as fp:
                record_number, airspace_list, mapspaces, checkspaces, bbox = openair_content_to_data(fp)
        except (TypeError, ValueError, Exception):
            # raise
            '''Try to correct content format'''
            try:
                fp.seek(0)
                content = ''
                for line in fp:
                    if not line.startswith('*'):
                        content += line.replace('  ', ' ')
                    else:
                        content += line
                fp.seek(0)
                fp.truncate()
                fp.write(content)
                fp.seek(0)
                record_number, airspace_list, mapspaces, checkspaces, bbox = openair_content_to_data(fp)
                modified = True
            except (TypeError, ValueError, Exception):
                '''Failure'''
                record_number = 0
        finally:
            fp.close()

        if record_number > 0:
            filename = unique_filename(file.filename, AIRSPACEDIR)
            save_airspace_map_check_files(filename, airspace_list, mapspaces, checkspaces, bbox)
            # save airspace file
            fullpath = Path(AIRSPACEDIR, filename)
            copyfile(tempfile, fullpath)

    return record_number, filename, modified


def get_igc_filename_formats_list() -> list:
    """ returns track filename formats that are automatically recognised when bulk importing through zip file"""
    return filename_formats


def check_participants_ids(comp_id: int, pilots: list) -> list:
    """gets a list of pilots and checks their ID validity against registered pilots and correct formats
    returns a list of pilots with correct IDs"""
    from pilot.participant import get_valid_ids

    return get_valid_ids(comp_id, pilots)


def check_zip_file(file: Path, extensions: list = None) -> tuple:
    """function to check if zip file is a valid archive and is not empty"""
    from zipfile import ZipFile, is_zipfile

    if not is_zipfile(file):
        return False, 'File is not a valid archive.'
    zipfile = ZipFile(file)
    if zipfile.testzip():
        return False, 'Zip file is corrupt.'
    elements = zipfile.namelist()
    if not elements or extensions and not any(el for el in elements if Path(el).suffix[1:].lower() in extensions):
        return False, f'Zip file is empty or does not contain any file with extension: {", ".join(extensions)}.'
    return True, 'success'


def get_comp_rankings(comp_id: int) -> list:
    from ranking import CompRanking
    rankings = []
    rows = CompRanking.read_all(comp_id)
    for el in rows:
        rankings.append(dict(description=el.description, **el.as_dict()))
    return rankings


def get_certifications_details() -> dict:
    from db.tables import TblCertification as CCT
    certifications = {}
    with db_session() as db:
        certs = db.query(CCT).order_by(CCT.comp_class, CCT.cert_order.desc())
        for cl in ('PG', 'HG'):
            certifications[cl] = [{'cert_id': el.cert_id, 'cert_name': el.cert_name}
                                  for el in certs if el.comp_class == cl]
    return certifications


def get_comp_meta(comp_id: int) -> list:
    from db.tables import TblCompAttribute as CA, TblCompRanking as CR
    with db_session() as db:
        results = db.query(CA).filter_by(comp_id=comp_id)
        ranks = [el.attr_id for el in db.query(CR).filter_by(comp_id=comp_id).distinct() if el.attr_id]
        return [{'attr_id': el.attr_id,
                 'attr_key': el.attr_key,
                 'attr_value': el.attr_value,
                 'used': True if el.attr_id in ranks else False} for el in results]


def add_custom_attribute(comp_id: int, attr_value: str) -> int or None:
    from ranking import CompAttribute
    attribute = CompAttribute(attr_key='meta', attr_value=attr_value, comp_id=comp_id)
    attribute.to_db()
    return attribute.attr_id


def edit_custom_attribute(data: dict) -> bool:
    from ranking import CompAttribute
    try:
        attribute = CompAttribute(attr_key='meta', **data)
        attribute.to_db()
        return True
    except Exception:
        return False


def remove_custom_attribute(attr_id: int) -> bool:
    from db.tables import TblCompAttribute as CA
    with db_session() as db:
        try:
            db.query(CA).get(attr_id).delete()
            return True
        except Exception:
            # raise
            return False


def save_airspace_check(comp_id: int, task_id: int, obj: dict) -> bool:
    """At the moment Airspace check is considering each event has just one parameters setup, but arguments are
    ready to create task settings. I don't think there is this need at the moment.
    task_id is always None"""
    from db.tables import TblAirspaceCheck as A
    try:
        row = A.get_one(comp_id=comp_id, task_id=task_id)
        row.update(**obj)
        return True
    except Exception:
        # raise
        return False


def start_livetracking(task_id: int, username: str, interval: int = 60):
    import rq
    if production():
        q = current_app.task_queue
        job_id = f'job_start_livetracking_task_{task_id}'
        job = q.enqueue(
            start_livetracking_background,
            args=(
                task_id,
                username,
                interval
            ),
            job_id=job_id,
            retry=rq.Retry(max=3),
            job_timeout=180
        )
        return job
    return None


def start_livetracking_background(task_id: int, username: str, interval: int):
    from livetracking import LiveTracking

    print = partial(print_to_sse, id=task_id, channel=username)

    lt = LiveTracking.read(task_id)
    if lt:
        task_name = lt.task.task_name
        if lt.filename.exists():
            lt.run()
            print(f'{task_name}: Livetracking Restored|livetracking')
        else:
            lt.create()
            print(f'{task_name}: Livetracking Started|livetracking')
        '''schedule livetracking after task window is open'''
        if lt.opening_timestamp > lt.timestamp + interval:
            delay = int(lt.opening_timestamp - lt.timestamp)
            call_livetracking_scheduling_endpoint(task_id, username, interval, delay)
        else:
            call_livetracking_scheduling_endpoint(task_id, username, interval)
    else:
        print(f'Error creating Livetracking (task ID {task_id}|livetracking')


def call_livetracking_scheduling_endpoint(task_id: int, username: str, interval: int, delay: int = 0):
    import time

    job_id = f'job_{int(time.time())}_livetracking_task_{task_id}'
    data = {'taskid': task_id, 'job_id': job_id, 'username': username, 'interval': interval, 'delay': delay}
    url = f"http://{environ.get('FLASK_CONTAINER')}:" f"{environ.get('FLASK_PORT')}/internal/_progress_livetrack"

    try:
        resp = requests.post(
            url,
            json=data,
            timeout=2
        )
    except requests.exceptions.ReadTimeout:
        return 'Timed Out'
    return job_id, resp.content


def call_livetracking_stopping_endpoint(task_id: int, username: str):

    data = {'taskid': task_id, 'username': username}
    url = f"http://{environ.get('FLASK_CONTAINER')}:" f"{environ.get('FLASK_PORT')}/internal/_stop_livetrack"

    try:
        resp = requests.post(
            url,
            json=data,
            timeout=2
        )
    except requests.exceptions.ReadTimeout:
        return 'Timed Out'
    return resp.content


def schedule_livetracking(task_id: int, job_id: str, username: str, interval: int = 60, delay: int = 0):
    import rq
    from datetime import timedelta
    if production():
        q = current_app.task_queue
        job = q.enqueue_in(
            timedelta(seconds=interval+delay),
            process_livetracking_background,
            args=(
                task_id,
                username,
                interval
            ),
            job_id=job_id,
            retry=rq.Retry(max=3),
            job_timeout=180
        )
        return job


def process_livetracking_background(task_id: int, username: str, interval: int):
    from livetracking import LiveTracking
    from rq import get_current_job
    print = partial(print_to_sse, id=task_id, channel=username)
    job_id = get_current_job().id
    lt = LiveTracking.read(task_id)

    if lt.properly_set:
        results = lt.run(interval)
        '''return final track results via sse'''
        for data in results:
            print(json.dumps(data) + '|result')

    print(f"{lt.task.task_name}: {lt.status}|livetracking")
    if lt.finished:
        print(f"{lt.task.task_name}: Stopping Livetrack|livetracking")
        lt.finalise()
        '''cancel livetracking schedules. Should not be needed'''
        call_livetracking_stopping_endpoint(task_id, username)
    else:
        '''schedule next livetracking run'''
        call_livetracking_scheduling_endpoint(task_id, username, interval)


def stop_livetracking(task_id: int, username: str):
    """ To stop livetracking, we stop currently working job and cancel scheduled job from queue"""
    from rq import cancel_job

    if production():
        q = current_app.task_queue
        sched = q.scheduled_job_registry
        failed = q.failed_job_registry
        job_id = f'livetracking_task_{task_id}'
        '''stopping running job'''
        for el in (j for j in sched.get_job_ids() if j.endswith(job_id)):
            cancel_job(el, current_app.redis)
        '''removing job from failed registry to avoid retry'''
        for el in (j for j in failed.get_job_ids() if j.endswith(job_id)):
            failed.remove(el, delete_job=True)
        '''removing job from scheduled registry and delete the job'''
        for el in (j for j in sched.get_job_ids() if j.endswith(job_id)):
            sched.remove(el, delete_job=True)
        return True


def create_new_comp(comp, user_id: int) -> dict:
    from compUtils import create_comp_path
    from formula import Formula, list_formulas
    from ranking import create_overall_ranking
    from airspace import create_check_parameters

    comp.comp_path = create_comp_path(comp.date_from, comp.comp_code)
    output = comp.to_db()
    if isinstance(output, int):
        formulas = list_formulas().get(comp.comp_class)
        formula = Formula.from_preset(comp.comp_class, formulas[-1])
        formula.comp_id = comp.comp_id
        formula.to_db()
        set_comp_scorekeeper(comp.comp_id, user_id, owner=True)
        create_overall_ranking(comp.comp_id)
        create_check_parameters(comp.comp_id)
        return {'success': True}
    return {'success': False, 'errors': {'Error': ['There was an error trying to save new Competition']}}


def recheck_track(task_id: int, par_id: int, user) -> tuple:
    from airspace import AirspaceCheck
    from pilot.flightresult import FlightResult, save_track
    from trackUtils import check_flight, igc_parsing_config_from_yaml, import_igc_file
    from task import Task

    if production():
        job = current_app.task_queue.enqueue(recheck_track_background,
                                             task_id, par_id, user)
        return True, None

    pilot = FlightResult.read(par_id=par_id, task_id=task_id)
    task = Task.read(task_id)

    """import track"""
    file = Path(task.file_path, pilot.track_file)
    FlightParsingConfig = igc_parsing_config_from_yaml('_overide')
    flight, error = import_igc_file(file, task, FlightParsingConfig)
    if error:
        return False, error['text']

    '''recheck track'''
    airspace = None if not task.airspace_check else AirspaceCheck.from_task(task)
    check_flight(pilot, flight, task, airspace, print=print)
    '''save to database'''
    save_track(pilot, task.id)

    print(f"track verified with task {task.task_id}\n")

    data = track_result_output(pilot, task_id)
    return data, None


def recheck_track_background(task_id: int, par_id: int, user):
    from trackUtils import import_igc_file, igc_parsing_config_from_yaml, check_flight
    import json
    from airspace import AirspaceCheck
    from pilot.flightresult import FlightResult, save_track
    from task import Task

    print = partial(print_to_sse, id=par_id, channel=user)
    print('|open_modal')

    pilot = FlightResult.read(par_id, task_id)
    task = Task.read(task_id)
    data = {'par_id': pilot.par_id, 'track_id': pilot.track_id}

    """import track"""
    file = Path(task.file_path, pilot.track_file)
    FlightParsingConfig = igc_parsing_config_from_yaml('_overide')
    flight, error = import_igc_file(file, task, FlightParsingConfig)
    if error:
        '''error importing igc file'''
        print(f"Error: {error['text']}")
        print(f"{json.dumps(data)}|{error['code']}")
        return None

    '''recheck track'''
    airspace = None if not task.airspace_check else AirspaceCheck.from_task(task)
    print('***************START*******************')
    check_flight(pilot, flight, task, airspace, print=print)
    '''save to database'''
    save_track(pilot, task.id)

    data = track_result_output(pilot, task.task_id)
    data['outdated'] = False

    print(json.dumps(data) + '|result')
    print('***************END****************')

    return True


def recheck_tracks(task_id: int, username: str):
    """get list of tracks that need to be evaluated, and process them"""
    from pilot.flightresult import FlightResult, update_all_results
    from task import Task
    from airspace import AirspaceCheck
    from trackUtils import igc_parsing_config_from_yaml, check_flight
    from pilot.track import Track

    if production():
        job = current_app.task_queue.enqueue(recheck_tracks_background,
                                             task_id=task_id,
                                             user=username,
                                             job_timeout=2000)
        return True

    par_ids = get_outdated_tracks(task_id)
    pilots_to_save = []
    task = Task.read(task_id)
    track_path = task.file_path
    FlightParsingConfig = igc_parsing_config_from_yaml(task.igc_config_file)
    airspace = None if not task.airspace_check else AirspaceCheck.from_task(task)

    for par in par_ids:
        pilot = FlightResult.read(par_id=par, task_id=task_id)
        file = Path(track_path, pilot.track_file)
        flight = Track.process(file, task, config=FlightParsingConfig)
        if flight:
            check_flight(pilot, flight, task, airspace=airspace)
            pilots_to_save.append(pilot)
    '''save all succesfully processed pilots to database'''
    update_all_results([p for p in pilots_to_save], task_id)
    return True


def recheck_tracks_background(task_id: int, user: str):
    from task import Task
    from airspace import AirspaceCheck
    from trackUtils import igc_parsing_config_from_yaml, check_flight
    from pilot.flightresult import update_all_results
    from pilot.track import Track

    pilots_to_save = []
    print = partial(print_to_sse, id=None, channel=user)
    print('|open_modal')
    task = Task.read(task_id)
    track_path = task.file_path
    task.get_pilots()
    par_ids = get_outdated_tracks(task_id)
    outdated_results = filter(lambda x: x.par_id in par_ids, task.results)
    FlightParsingConfig = igc_parsing_config_from_yaml(task.igc_config_file)
    airspace = None if not task.airspace_check else AirspaceCheck.from_task(task)
    for pilot in outdated_results:
        # pilot = FlightResult.read(par_id=par, task_id=task_id)
        file = Path(track_path, pilot.track_file)
        flight = Track.process(file, task, config=FlightParsingConfig)
        print(f"processing {pilot.ID} {pilot.name}:")
        if not flight:
            print('Error: IGC file not readable')
        else:
            pilot_print = partial(print_to_sse, id=pilot.par_id, channel=user)
            print('***************START*******************')
            check_flight(pilot, flight, task, airspace, print=pilot_print)
            if pilot.notifications:
                print(f"NOTES:<br /> {'<br />'.join(n.comment for n in pilot.notifications)}")

            pilots_to_save.append(pilot)
            data = track_result_output(pilot, task.task_id)
            pilot_print(f'{json.dumps(data)}|result')
            print('***************END****************')
    print("*****************re-processed all tracks********************")

    '''save all succesfully processed pilots to database'''
    update_all_results([p for p in pilots_to_save], task_id)
    print('|page_reload')


def recheck_needed(task_id: int):
    from task import task_need_recheck
    return task_need_recheck(task_id)

