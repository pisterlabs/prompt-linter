import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

MAP = [
    "+----+",
    "|   G|",
    "| | F|",
    "|S   |",
    "+----+"
]

class MDPGridworldEnv(discrete.DiscreteEnv):
    """
    IMPORTANT: This code is based from openAI gym's FrozenLake code.
    This is a 3x4 grid world based from problem for an AI-Class. (https://goo.gl/GqkyzT)
    The surface is described using a grid like the following

    S : starting point, non-terminal state (reward: -3)
      : non-terminal states (reward: -3)
    F : fire, burn to death (reward: -100)
    G : goal (reward: +100)

    The episode ends when you reach the goal or burn in hell.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[3,4]):
        self.shape = shape
        self.desc = np.asarray(MAP,dtype='c')

        self.nrow = nR = 3
        self.ncol = nC = 4
        nA = 4
        nS = nR * nC

        isd = np.zeros(nS)


        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*self.ncol + col
        def inc(row, col, a):
            t_row, t_col = row, col
            if a==WEST:
                col = max(col-1,0)
            elif a==SOUTH:
                row = min(row+1,self.nrow-1)
            elif a==EAST:
                col = min(col+1,self.ncol-1)
            elif a==NORTH:
                row = max(row-1,0)
            if self.desc[row+1,col+1] == b'|':
                row, col = t_row, t_col
            return (row, col)

        isd[to_s(2, 0)] = 1
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = self.desc[row+1, col+1]
                    if letter in b'GF':
                        li.append((1.0, s, 0, True))
                    elif letter == b'|':
                        li.append((0.0, s, 0, False))
                    else:
                        rew = -3
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = self.desc[newrow+1, newcol+1]
                        done = bytes(newletter) in b'GF'
                        if bytes(newletter) in b'G':
                            rew = 100
                        elif bytes(newletter) in b'F':
                            rew = -100
                        li.append((1.0, newstate, rew, done))

        isd /= isd.sum()
        super(MDPGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = (self.s // self.ncol)+1, (self.s % self.ncol)+1
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["North","South","West","East"][self.lastaction]))
        else:
            outfile.write("\n")

        return outfile
