# Copyright 2020 EcoSys Group Sdn Bhd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import Sequence
import sqlite3
from sqlite3 import Error
import numpy as np
import io
import json
import time
from openai_ros2.tasks.lobot_arm.arm_random_goal import ArmState


def adapt_np_array(arr: np.ndarray):
    return json.dumps(arr.tolist())


def convert_np_array(text):
    return np.array(json.loads(text))


def adapt_arm_state(state: ArmState):
    return str(state.name)


def convert_arm_state(state):
    key = state.decode('utf-8')
    return ArmState[key]


class Logger:
    def __init__(self, table_name, db_file: str = 'env_log.db'):
        self.table_name = table_name
        """ create a database connection to a SQLite database """
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
            print('Using sqlite3, version: {sqlite3.version}')
        except Error as e:
            print(e)

        if self.conn is not None:
            # TODO: unique table name, since we know the datetime of the data anyway
            create_success = self.create_table(self.table_name)
            if not create_success:
                print(f'Failed to create table: {self.table_name}')
                self.has_table = False
            else:
                self.has_table = True
        sqlite3.register_adapter(np.ndarray, adapt_np_array)
        sqlite3.register_converter("np_array", convert_np_array)
        sqlite3.register_adapter(ArmState, adapt_arm_state)
        sqlite3.register_converter("armstate", convert_arm_state)

        self.store_count = 0
        self.buffer_size = 20000
        self.buffer = []

    def create_table(self, table_name) -> bool:
        # Currently the SQL is hardcoded, maybe can consider autogenerate if the table requirements change a lot
        # Few custom types are used,
        # 1. ArmState (the enum)
        # 2. np.ndarray (for serialization and deserialization of numpy arrays into json text)
        c = self.conn.cursor()

        # We drop the table if it exists (it shouldn't)
        drop_table_sql = """DROP TABLE %s""" % table_name
        try:
            c.execute(drop_table_sql)
        except Error as e:
            print(e)

        print(f'Creating table: {table_name}')
        create_table_sql = """CREATE TABLE %s (
                                        id integer PRIMARY KEY,
                                        episode_num integer NOT NULL,
                                        step_num integer NOT NULL,
                                        arm_state armstate NOT NULL,
                                        dist_to_goal real NOT NULL,
                                        target_coords np_array NOT NULL,
                                        current_coords np_array NOT NULL,
                                        joint_pos np_array NOT NULL,
                                        joint_vel np_array NOT NULL,
                                        rew_noise real NOT NULL,
                                        reward real NOT NULL,
                                        normalised_reward real NOT NULL,
                                        exp_reward real NOT NULL,
                                        cum_unshaped_reward real NOT NULL,
                                        cum_normalised_reward real NOT NULL,
                                        cum_exp_reward real NOT NULL,
                                        cum_reward real NOT NULL,
                                        cum_rew_noise real NOT NULL,
                                        action np_array NOT NULL,
                                        date_time text NOT NULL
                                    );""" % self.table_name
        try:
            c.execute(create_table_sql)
            return True
        except Error as e:
            print(e)
            return False

    def __del__(self):
        self.__store_buffer_into_db()

    def store(self, episode_num: int, step_num: int, arm_state: ArmState, dist_to_goal: float, target_coords: np.ndarray, current_coords: np.ndarray,
              joint_pos: np.ndarray, joint_pos_true: np.ndarray, joint_vel: np.ndarray, joint_vel_true: np.ndarray, rew_noise: float,
              reward: float, normalised_reward: float, exp_reward: float, cum_unshaped_reward: float, cum_normalised_reward: float,
              cum_exp_reward:float, cum_reward: float, cum_rew_noise: float,
              action: np.ndarray):
        # if not self.has_table:
        #     print('No table created. Not logging')
        #     return
        assert self.has_table
        assert isinstance(step_num, int)
        assert isinstance(episode_num, int)
        assert isinstance(arm_state, ArmState)
        assert isinstance(dist_to_goal, float)
        assert isinstance(target_coords, np.ndarray)
        assert isinstance(current_coords, np.ndarray)
        assert isinstance(joint_pos, np.ndarray)
        # assert isinstance(joint_pos_true, np.ndarray)
        assert isinstance(joint_vel, np.ndarray)
        # assert isinstance(joint_vel_true, np.ndarray)
        assert isinstance(rew_noise, float)
        assert isinstance(reward, float)
        assert isinstance(normalised_reward, float)
        assert isinstance(exp_reward, float)
        assert isinstance(cum_unshaped_reward, float)
        assert isinstance(cum_normalised_reward, float)
        assert isinstance(cum_exp_reward, float)
        assert isinstance(cum_reward, float)
        assert isinstance(cum_rew_noise, float)
        assert isinstance(episode_num, int)
        assert isinstance(action, np.ndarray)

        # TODO: Store datetime into this tuple and store it into the database instead of letting the database generate datetime
        data_tup = (episode_num, step_num, arm_state, dist_to_goal, target_coords, current_coords, joint_pos, joint_vel,
                    rew_noise, reward, normalised_reward, exp_reward, cum_unshaped_reward, cum_normalised_reward, cum_exp_reward,
                    cum_reward, cum_rew_noise, action)
        self.buffer.append(data_tup)
        if len(self.buffer) >= self.buffer_size:
            self.__store_buffer_into_db()

    def load(self, sql_statement: str = None):
        if len(self.buffer) > 0:
            self.__store_buffer_into_db()

        cur = self.conn.cursor()
        if sql_statement is None:
            cur.execute(f"select * from {self.table_name}")
            data = cur.fetchall()
            return data
        else:
            cur.execute(sql_statement)
            data = cur.fetchall()
            return data

    def get_table_name(self) -> str:
        return self.table_name

    def __store_buffer_into_db(self):
        cur = self.conn.cursor()
        for data in self.buffer:
            cur.execute(f"insert into {self.table_name} (episode_num, step_num, arm_state, dist_to_goal, target_coords,"
                        f"current_coords, joint_pos, joint_vel,"
                        f"rew_noise, reward, normalised_reward, exp_reward, cum_unshaped_reward, cum_normalised_reward, cum_exp_reward,"
                        f"cum_reward, cum_rew_noise, action, date_time) "
                        f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now', 'localtime'))", data)
        self.conn.commit()
        self.buffer = []
