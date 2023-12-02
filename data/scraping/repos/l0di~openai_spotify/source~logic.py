from enum import Enum
from loglib import logger
import argparse
import chat
import cohere_lib
import os
import spotify
import sys
import twilio_lib
import ttdb

import signal
import secrets
import threading
import queue
import time
import random

class ERROR_CODES(Enum):
  NO_ERROR=0
  ERROR_NO_SPOTIFY_USER=1
  ERROR_CHAT_GPT=2
  ERROR_NO_SPOTIFY_RECS=3
  ERROR_NO_PLAYLIST_CREATE=4
  ERROR_NO_GEN=5
  ERROR_COHERE_GEN=6
  ERROR_SPOTIFY_REFRESH=7


def handler(signum, frame):
  raise Exception("Timeout!")
signal.signal(signal.SIGALRM, handler)


def get_playlist_attributes_cohere(user_query, attrs=None, genres=None):
  artist_prompt = chat.create_artist_prompt(user_query)
  song_prompt = chat.create_song_prompt(user_query)
  genre_prompt = chat.create_genre_prompt(user_query, genres)
  playlist_prompt = chat.create_playlist_prompt(user_query)
  attr_prompt = chat.create_attribute_prompt(user_query, attrs)
  tempo_prompt = chat.create_tempo_prompt(user_query)
  prompts = {
    'artists': {'prompt': artist_prompt, 'temp': .3, 'model': 'command', 'p': .9},
    'songs': {'prompt': song_prompt, 'temp': .3, 'model': 'command', 'p': .9},
    'genres': {'prompt': genre_prompt, 'temp': .6, 'model': 'command', 'p': .9},
    'playlist': {'prompt': playlist_prompt, 'temp': .8,
          'model': 'command', 'p': .9, 'tokens': 100},
    'attrs': {'prompt': attr_prompt, 'temp': .5, 'model': 'command', 'p': .9},
    'tempo': {'prompt': tempo_prompt, 'temp': .5, 'model': 'command', 'p': .9},
    }
  
  q = queue.Queue()
  def _cohere_thread(ptype, vals):
    prompt = vals['prompt']
    temp = vals['temp']
    model = vals.get('model', 'command-light')
    p = vals.get('p', .75)
    tokens = vals.get('tokens', 75)
    try:
      text = cohere_lib.get_assistant_message_with_str(prompt,
        max_tokens=tokens, temperature=temp, model=model, p=p)
    except Exception as e:
      logger.info('cohere thread exception: %s', e)
      return

    if text is None:
      text = ''
    text = text.strip()
    logger.info('cohere single: %s: %s', ptype, text)
    if ptype == 'attrs':
      att = attrs
    elif ptype == 'tempo':
      att = ['tempo']
    else:
      att = []
    want = spotify.chatOutputToStructured(f'{ptype}: {text}',
            attributes=att, want=ptype)
    q.put({ptype: want})

  threads = []
  for k, v in prompts.items():
    t = threading.Thread(target=_cohere_thread, args=(k, v))
    threads.append(t)

  for t in threads:
    t.start()
  for t in threads:
    t.join()

  outputs = {
    'artists': [],
    'songs': {},
    'genres': [],
    'playlist': [],
    'attrs': {},
    'tempo': {},
  }
  while not q.empty():
    outputs.update(q.get())

  if not any(list(outputs.values())):
    return ERROR_CODES.ERROR_NO_GEN, None

  for k, v in outputs.items():
    logger.info('Cohere output: %s: %s', k, v)

  return outputs


def get_playlist_attributes(msgs, attributes, number_id=None):
  q = queue.Queue()
  def oai_chat_thread(msgs, attributes, number_id):
    chat_outputs = chat.get_assistant_message(msgs, number_id=number_id)
    if not chat_outputs:
      return ERROR_CODES.ERROR_CHAT_GPT, None
    logger.info('%s: openai output: %s', number_id, chat_outputs)
    s_genres, s_artists, s_songs, s_attrs, s_playlist = spotify.chatOutputToStructured(
        chat_outputs, attributes=attributes, number_id=number_id)
    vals = {
      'genres': s_genres, 'artists': s_artists,
      'songs': s_songs, 'attrs': s_attrs,
      'playlist': s_playlist
      }
    q.put({'oai': vals})

  def cohere_chat_thread(msgs, attributes, number_id):
    chat_outputs = cohere_lib.get_assistant_message(msgs, number_id=number_id)
    logger.info('%s: cohere output: %s', number_id, chat_outputs)
    s_genres, s_artists, s_songs, s_attrs, s_playlist = spotify.chatOutputToStructured(
      chat_outputs, attributes=attributes, number_id=number_id)
    vals = {
      'genres': s_genres, 'artists': s_artists,
      'songs': s_songs, 'attrs': s_attrs,
      'playlist': s_playlist
      }
    q.put({'cohere': vals})
  
  t_oai = threading.Thread(target=oai_chat_thread, args=(msgs, attributes, number_id))
  t_cohere = threading.Thread(target=cohere_chat_thread, args=(msgs, attributes, number_id))
  threads = [t_oai, t_cohere]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  outputs = []
  while not q.empty():
    outputs.append(q.get())
  
  oai_vals = [k['oai'] for k in outputs if 'oai' in k]
  oai_vals = oai_vals[0] if oai_vals else []
  logger.info('OAI vals: %s', oai_vals)
  cohere_vals = [k['cohere'] for k in outputs if 'cohere' in k]
  cohere_vals = cohere_vals[0] if cohere_vals else []
  logger.info('cohere vals: ', cohere_vals)
   
  using_cohere = False
  if not any(list(oai_vals.values())):
    if not any(list(cohere_vals.values())):
      return ERROR_CODES.ERROR_NO_GEN, None
    using_cohere = True
    return using_cohere, cohere_vals
  return using_cohere, oai_vals


def get_spotify_song_artists(spot, s_artists, s_songs):
  q = queue.Queue()
  def artist_thread(spot, artists):
    q.put({'artists': spot.IdsForArtists(artists)})

  def song_thread(spot, songs):
    q.put({'songs': spot.IdsForSongs(songs)})

  t_artist = threading.Thread(target=artist_thread, args=(spot, s_artists)) 
  t_song = threading.Thread(target=song_thread, args=(spot, s_songs)) 
  threads = [t_artist, t_song]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  outputs = []
  while not q.empty():
    outputs.append(q.get())

  songs = [k['songs'] for k in outputs if 'songs' in k]
  songs = songs[0] if songs else []
  artists = [k['artists'] for k in outputs if 'artists' in k]
  artists = artists[0] if artists else []
  return artists, songs


def get_playlist_name(user_query, using_cohere, number_id):
  with_retry = False
  pname = None
  msgs = chat.create_playlist_name_from_query(user_query, with_retry=with_retry)
  if not using_cohere:
    pnames_list = chat.get_assistant_message(msgs, temperature=.5, number_id=number_id)
  else:
    pnames_list = cohere_lib.get_assistant_message(msgs, max_tokens=100, number_id=number_id)
  logger.info('%s: llm output playlist list: %s', number_id, pnames_list)
  names = [name for name in chat.parse_playlist_name(pnames_list)]
  return secrets.choice(names)


def find_playlist_name(s_playlist_names: list):
  """Find a playlist name that is not used and add to db."""
  pname = None
  for name in s_playlist_names:
    if not twilio_lib.db.playlist_name_exists(name):
      pname = name
      # insert name into db
      spot_plist = ttdb.SpotifyPlaylistNames(name=pname)
      twilio_lib.db.playlist_name_insert(spot_plist.dict())
      return pname
 
  playlist_url = None
  playlist_id = None
  if pname is None:
    for name in s_playlist_names:
      for i in range(10):
        cur_name = ''.join(random.choice((str.upper, str.lower))(c) for c in name)
        if not twilio_lib.db.playlist_name_exists(cur_name):
          pname = cur_name
          # insert name into db
          spot_plist = ttdb.SpotifyPlaylistNames(name=pname)
          twilio_lib.db.playlist_name_insert(spot_plist.dict())
          return pname

  return None


def playlist_for_query(user_query: str,
    number_id: str,
    access_token: str = '',
    refresh_token: str = '',
    include_all_playlist_info: bool = False
    ):
  """Responds with tuple of (Error, Message)."""
  og_query = user_query
  user_query = 'Make me a musical playlist that conforms to: ' + user_query
  spot = spotify.SpotifyRequest()
  if access_token:
    using_thumb_tings = False
    access_token = spotify.spotify_refresh_token()
    spot.token = access_token
    cuser = spot.current_user()
    logger.info('cuser: %s', cuser)
    # force refresh
    if cuser is None:
      # always backup to using ThumbTings
      spot.reinit()
      using_thumb_tings = True
      logger.info("Using thumbtings account")
      cuser = spot.current_user()
      logger.info('cuser: %s', cuser)
      if not cuser:
        return ERROR_CODES.ERROR_NO_SPOTIFY_USER, None

    if not using_thumb_tings:
      spot._username = cuser['id']

      screds = ttdb.SpotifyCreds(
        username=cuser['id'],
        access_token=access_token,
        refresh_token=refresh_token
      )
      db = twilio_lib.db  
      if not db.spotify_user_exists(cuser['id']):
        db.spotify_insert(screds.dict())
      else:
        db.spotify_update_user(access_token, refresh_token, cuser['id'])

      logger.info('current spotify user: %s', cuser['id'])
  else:
    spot.reinit()
    logger.info('cuser: %s', spot.current_user())
    if not spot.current_user():
      return ERROR_CODES.ERROR_NO_SPOTIFY_USER, None
  
  spot.userCanSearch()

  genres = spot.get_genre_seeds()['genres']
  attributes = spot.get_attributes()
  # use cohere by default
  if 1: # TODO (cohere sucks)
    using_cohere = True
    chat_vals = get_playlist_attributes_cohere(user_query, attrs=attributes, genres=genres) 
  else:
    msgs = chat.create_prompt(user_query, attrs=attributes, genres=genres)
    using_cohere, chat_vals = get_playlist_attributes(msgs, attributes,
                                                      number_id=number_id)
  if chat_vals is None:
    return ERROR_CODES.ERROR_NO_SPOTIFY_RECS, None

  s_genres = chat_vals['genres']
  s_artists = chat_vals['artists']
  s_songs = chat_vals['songs']
  s_attrs = chat_vals['attrs']
  s_attrs.update(chat_vals['tempo'])
  s_playlist_names = chat_vals['playlist']

  s_artists = list(set(s_artists + list(s_songs.values())))
  logger.info('%s: Guessed genres: %s', number_id, s_genres)
  s_genres = [g for g in s_genres if g in genres]
  logger.info('%s: Spotify genres: %s', number_id, s_genres)
  logger.info('%s: Spotify artists: %s', number_id, s_artists)
  logger.info('%s: Spotify songs: %s', number_id, s_songs)
  logger.info('%s: Spotify attributes: %s', number_id, s_attrs)
  logger.info('%s: Spotify playlist name: %s', number_id, s_playlist_names)
  logger.info('using cohere: %s', using_cohere)

  s_artists, s_songs = get_spotify_song_artists(spot, s_artists, s_songs)

  logger.info('%s: found artists: %s', number_id, s_artists)
  logger.info('%s: found songs: %s', number_id, s_songs)

  recs = spot.get_recommendations(seed_genres=s_genres, seed_artists=s_artists, 
    seed_tracks=s_songs, attributes=s_attrs)
  if not recs:
    logger.info('%s: no recs found', number_id)
    return ERROR_CODES.ERROR_NO_SPOTIFY_RECS, None
  track_uris = spot.tracksForRecs(recs)
  logger.info('%s: track uris length: %s', number_id, len(track_uris))

  pname = find_playlist_name(s_playlist_names)
  if pname is None:
    logger.info('%s: playlist name is none', number_id)
    return ERROR_CODES.ERROR_NO_PLAYLIST_CREATE, None

  playlist_id, playlist_url = spot.create_playlist(pname=pname, description=og_query)

  if playlist_id is None:
    logger.info('%s: playlist id is none', number_id)
    return ERROR_CODES.ERROR_NO_PLAYLIST_CREATE, None
  spot.playlist_write_tracks(playlist_id, track_uris)

  logger.info('%s: playlist url: %s', number_id, playlist_url)
  if access_token or include_all_playlist_info:
    playlist_image = spot.playlist_cover_image(playlist_id)
    return ERROR_CODES.NO_ERROR, (playlist_url, playlist_image)
  else:
    return ERROR_CODES.NO_ERROR, playlist_url



