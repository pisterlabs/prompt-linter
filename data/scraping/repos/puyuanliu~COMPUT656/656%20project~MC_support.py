
# -----------------------------------------------------------
# Support function & classes for Extended Mountain Car problem.
# Extended Mountain Car problem has a different goal compared to the usual
# Mountain Car problem, that is, stop at a target height instead of getting
# out of the bottom.
# Tile coding related functions come from Tile3 package wrote by Rich Sutton.
# Tile coder class come from Calarina Muslimani
# Extended Mountain Car environment is modified based on the Mountain Car
# environment of Gym package from OpenAI.
#
# (C) 2020 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# email puyuan@ualberta.ca
# -----------------------------------------------------------



import numpy as np
from math import floor, log
from itertools import zip_longest


class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + " size:" + str(self.size) + " overfullCount:" + str(
            self.overfullCount) + " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount == 0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hashcoords(coordinates, m, readonly=False):
    if type(m) == IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m) == int: return hash(tuple(coordinates)) % m
    if m == None: return coordinates

def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b % numtilings) // numtilings
            coords.append(c % width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


class ExtendedMountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
            Initializes the Extended MountainCar Tile Coder

            iht_size -- int, the size of the index hash table, typically a power of 2
            num_tilings -- int, the number of tilings
            num_tiles -- int, the number of tiles. Here both the width and height of the
            tile coder are the same
            """
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self, position, velocity):
        """
            Takes in a position and velocity from the mountaincar environment
            and returns a numpy array of active tiles.

            returns:
            tiles - np.array, active tiles
            """
        # Use the ranges above and self.num_tiles to scale position and velocity to the range [0, 1]
        # then multiply that range with self.num_tiles so it scales from [0, num_tiles]
        minP = -1.4
        maxP = .4
        minV = -.07
        maxV = .07
        scaleP = maxP - minP
        scaleV = maxV - minV

        position_scaled = ((position - minP) / (scaleP)) * self.num_tiles

        velocity_scaled = ((velocity - minV) / (scaleV)) * self.num_tiles

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        mytiles = tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])

        return np.array(mytiles)


