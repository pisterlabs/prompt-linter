import openai
import requests
import sqlite3

def connect():
    conn = sqlite3.connect("discord.db")
    cursor = conn.cursor()
    return conn, cursor
conn, cursor = connect()

def get_key(id_server):
    cursor.execute(f"SELECT token FROM token WHERE server_id = {id_server}")
    row = cursor.fetchone()
    conn.commit()
    if row == None:
        return 'no key'
    else:
        return row #Обожаю тебя Анечка)
        
def del_key(invalid_key):
    cursor.execute(f"DELETE FROM token WHERE token = '{invalid_key}'")
    conn.commit()
    
def add_key(server_id, keys):
    try:
        cursor.execute(f"INSERT INTO token(server_id, token) VALUES ({server_id}, '{keys}')")
        conn.commit()
        return 'good'
    except Exception as er:
        return er
        
def set_status(server_id, status):
    try:
        cursor.execute(f"SELECT tts_status FROM tts WHERE server_id = {server_id}")
        row = cursor.fetchone()
        conn.commit()
        if row[0] == f'{status}':
            return 'not required'
        elif row == None:
            cursor.execute(f"INSERT INTO tts(server_id) VALUES ({server_id})")
            conn.commit()
            cursor.execute(f"UPDATE tts SET tts_status = '{status}' WHERE server_id = {server_id}")
            conn.commit()
            return 'good'
        else:
            cursor.execute(f"UPDATE tts SET tts_status = '{status}' WHERE server_id = {server_id}")
            conn.commit()
            return 'good'
    except Exception as er:
        return er
        
def get_tts_status(server_id):
    try:
        cursor.execute(f"SELECT tts_status FROM tts WHERE server_id = {server_id}")
        row = cursor.fetchone()
        conn.commit()
        if row[0] == 'on':
            return 'on'
        elif row[0] == 'off':
            return 'off'
        elif row == None:
            cursor.execute(f"INSERT INTO tts(server_id) VALUES ({server_id})")
            conn.commit()
            return 'off'
    except Exception as er:
        return 'error'