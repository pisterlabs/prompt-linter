from itertools import product
import time

characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
              "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

all_combonations = characters.copy()

while len(all_combonations) > 0:
    charCombo = all_combonations.pop(-1)
    if charCombo == "0" or charCombo == "1" or charCombo == "0a" or charCombo == "1z":
        all_combonations += ["".join(pair) for pair in product([charCombo], characters)]

    print(all_combonations)    