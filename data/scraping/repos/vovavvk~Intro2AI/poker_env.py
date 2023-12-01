from random import random

import gym

from entities import *


class PokerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_of_chips, lose_punishment, deal_cards=True, randomize_chips=None):
        self.gm = None
        self.lp = lose_punishment
        self.dc = deal_cards
        self.rc = randomize_chips
        self.ob = self.reset(num_of_chips)
        self.fld_pen = 50
        self.win_rew = 100
        self.los_pen = 0
        self.tot_los_pen = 0
        self.tot_win_rew = 100

    def step(self, action):
        """Step

        returns ob, rewards, episode_over, info : tuple

        ob (list[Player,Player]) : next stage observation
            [player who is small blind,player who is big blind].
        rewards ([float,float]) :
            amount of reward achieved by the previous action
        :param action: [small blind action, big blind action]
                0 if fold, 1 otherwise
         """
        sb_player = self.gm.sb_player()
        bb_player = self.gm.bb_player()
        if action[0] == 1:
            # Small blind went all in
            self.gm.player_all_in(sb_player)
            if action[1] == 1:
                # BB called
                self.gm.player_call(bb_player, sb_player.bet)
                # Need to compare hands
                for c in range(5):
                    new_card = self.gm.deck.draw_card()
                    for ep in self.gm.p:
                        ep.hand.add_card(new_card)
                for pl in self.gm.p:
                    pl.hand.sort()
                # get the absolute score of the hand and the best five cards
                results = []
                for ep in [sb_player, bb_player]:
                    results.append(Game.score(ep.hand))
                # select the winner
                winners = Game.determine_winner(results)
                # award the pot to the winner
                if winners.__len__() > 1:
                    # split the pot
                    self.gm.split_the_pot()
                    # Give both players small reward
                    rewards = [self.win_rew, self.win_rew]
                else:
                    # Actually transfer the chips between players
                    _players = [sb_player, bb_player]
                    wnr = _players[winners[0]]
                    lsr = _players[1 - winners[0]]
                    self.gm.player_won(wnr)
                    rewards = []
                    # Reward the winner with chips won
                    rewards.insert(winners[0], self.win_rew)
                    # Penalty the loser with chips lost
                    rewards.insert(1 - winners[0], self.los_pen)
            else:
                # BB folded
                # Reward SB with amount won
                # Transfer chips to SB
                sb_player.bank += self.gm.pot
                # Penalty BB by amount lost
                rewards = [self.win_rew, self.fld_pen]
        else:
            # Small blind folded
            # Reward BB with 0 since their move didn't matter
            # Penalty SB by amount lost
            rewards = [self.fld_pen, self.win_rew]
            # Transfer chips to BB
            bb_player.bank += self.gm.pot
        # Change who is SB
        self.gm.sb = 1 - self.gm.sb
        self.gm.new_step()
        self.gm.place_blinds()
        if self.dc:
            self.gm.players_draw_cards()
        if self.gm.a_player().bank <= 0 or self.gm.na_player().bank <= 0:
            self.gm.done = True
        # if the game was lost completely, the punishment is different
        if self.gm.done:
            for (ind, r) in enumerate(rewards):
                if (r == self.los_pen) | (r == self.fld_pen):
                    rewards[ind] = self.tot_los_pen
        self.ob = [sb_player, bb_player]
        return self.ob, rewards, self.gm.done

    def reset(self, bank=50):
        """Initial setup for training

        :param bank : initial bank size
        :returns [small blind player, big blind player]
        """
        self.gm = Game("P1", "Qtable", "P2", "Qtable", bank)

        # Random starting money distribution p=0.4
        if self.rc is not None:
            if random() < self.rc:
                left = round(random() * ((bank * 2) - 4))
                self.gm.sb_player().bank = left + 2
                self.gm.bb_player().bank = ((bank * 2) - 4) - left + 2

        self.gm.place_blinds()
        if self.dc:
            self.gm.players_draw_cards()
        # Return observation
        # [ Small Blind Player, Big Blind Player ]
        self.ob = [self.gm.sb_player(), self.gm.bb_player()]
        return self.ob

    def render(self, mode='human'):
        self.gm.render_game()

    @staticmethod
    def encode(hand, small_blind, _num_of_chips, initial_num_of_chips):
        """Encoding and decoding code was lifted from openAI taxi gym"""
        # Sort hand and extract cards
        _sorted_hand = sorted(hand.cards, reverse=True)
        _card1 = _sorted_hand[0].encode()
        _card2 = _sorted_hand[1].encode()
        # Calculate coefficient of number of chips
        _coef_chips = 2 * _num_of_chips // initial_num_of_chips
        # Encode
        encoded = _card2
        encoded *= 52
        encoded += _card1
        encoded *= 2
        encoded += small_blind
        encoded *= 4
        encoded += _coef_chips
        return encoded

    @staticmethod
    def decode(_code):
        """Encoding and decoding code was lifted from openAI taxi gym"""
        _out = [_code % 4]
        _code = _code // 4
        _out.append(_code % 2)
        _code = _code // 2
        _card1 = Card.decode(_code % 52)
        _code = _code // 52
        _card2 = Card.decode(_code)
        assert 0 <= _code < 52
        _hand = Hand()
        _hand.add_card(_card1)
        _hand.add_card(_card2)
        _out.append(_hand)
        return _out


def test_encoder_decoder():
    """Test encoder and decoder"""
    hand1 = Hand()
    initial_chips = 10

    for chips in range(initial_chips * 2):
        for sb in range(2):
            for card1_code in range(52):
                for card2_code in range(52):
                    if card1_code == card2_code:
                        continue
                    # Create hand
                    card1 = Card.decode(card1_code)
                    card2 = Card.decode(card2_code)
                    hand1.clear_hand()
                    hand1.add_card(card1)
                    hand1.add_card(card2)
                    hand1.cards = sorted(hand1.cards, reverse=True)
                    # Encode and decode
                    code = PokerEnv.encode(hand1, sb, chips, initial_chips)
                    [new_chips, new_small_blind, new_hand] = PokerEnv.decode(code)
                    assert new_chips == 2 * chips // initial_chips
                    assert new_small_blind == sb
                    assert new_hand.cards[0] == hand1.cards[0]
                    assert new_hand.cards[1] == hand1.cards[1]


if __name__ == "__main__":
    # test_encoder_decoder()
    pass
