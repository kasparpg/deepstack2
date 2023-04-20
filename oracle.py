import random
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from helper_functions import cartesian_product, input_number
from itertools import combinations_with_replacement


class Card:
    def __init__(self, value: int, color: str):
        self.value = value
        self.color = color


class Player:
    def __init__(self, name: str, human: bool, cards: [], folded, role: str, chips: int, chips_added_to_table: int,
                 action_taken: bool):
        self.name = name
        self.human = human
        self.cards = cards
        self.folded = folded
        self.role = role
        self.chips = chips
        self.chips_added_to_table = chips_added_to_table
        self.action_taken = action_taken
        self.side_potted = False


# Check the winner.
def check_winner(players: [], cards_on_table: []):
    best_player_index = 0
    for i in range(len(players)):
        if i < len(players) - 1:
            player = players[best_player_index]
            next_player = players[i + 1]

            all_cards = []
            for card in player.cards + cards_on_table:
                all_cards.append(str(card.value)+card.color[0].upper())

            all_cards_next_player = []
            for card in next_player.cards + cards_on_table:
                all_cards_next_player.append(str(card.value)+card.color[0].upper())

            winner = compare_hands(all_cards, all_cards_next_player)

            if winner[0] == 'right':
                best_player_index = i + 1

    return players[best_player_index]


def generate_cheat_sheet(hand: [], cards_on_table: [], num_players: int, rollouts: int):
    print("\nGenerating cheat sheet...")
    # Get the hand cards in the correct format to compare hands.
    hand = np.array(correct_format(hand))

    # Get the deck cards in correct format to compare hands.
    deck = np.array(correct_format(shuffle_deck(create_deck(True), 100)))

    # Remove the cards in your hand from our created deck.
    deck = np.delete(deck, np.ravel([np.where(deck == i) for i in hand]))

    # Keep track of how often our hand wins.
    times_won = 0

    # Pre-flop
    if len(cards_on_table) == 0:
        # Loop through rollouts to calculate win %.
        for i in range(rollouts):
            # Seven random cards to simulate a game. The first 5 are the cards on the table, last 2 opponent's cards.
            random_cards = np.array(np.random.choice(deck, size=7, replace=False))
            # Check winner.
            if compare_hands(list(np.concatenate((hand, random_cards[0:5]))), list(random_cards))[0] == 'left':
                times_won += 1

    # Flop, Turn, River
    else:
        # Get the cards on the table to the correct format to compare hands.
        cards_on_table = np.array(correct_format(cards_on_table))

        # Remove the cards on the table from our created deck.
        deck = np.delete(deck, np.ravel([np.where(deck == i) for i in cards_on_table]))

        # Flop
        if len(cards_on_table) == 3:
            # Loop through rollouts to calculate win %.
            for i in range(rollouts):
                # Four random cards to simulate a game. The first 2 are the cards on the table, last 2 opponent's cards.
                random_cards = np.array(np.random.choice(deck, size=4, replace=False))
                # Check winner.
                if compare_hands(list(np.concatenate((hand, np.concatenate((random_cards[0:2], cards_on_table))))),
                                 list(np.concatenate((random_cards, cards_on_table))))[0] == 'left':
                    times_won += 1
        # Turn
        if len(cards_on_table) == 4:
            # Loop through rollouts to calculate win %.
            for i in range(rollouts):
                # Three random cards to simulate a game. The first 1 is the card on the table, last 2 opponent's cards.
                random_cards = np.array(np.random.choice(deck, size=3, replace=False))
                # Check winner.
                if compare_hands(list(np.concatenate((hand, np.concatenate((random_cards[0:1], cards_on_table))))),
                                 list(np.concatenate((random_cards, cards_on_table))))[0] == 'left':
                    times_won += 1
        # River
        if len(cards_on_table) == 5:
            for i in range(rollouts):
                # Two random cards to simulate a game. 2 opponent's cards.
                random_cards = np.array(np.random.choice(deck, size=2, replace=False))
                # Check winner.
                if compare_hands(list(np.concatenate((hand, cards_on_table))),
                                 list(np.concatenate((random_cards, cards_on_table))))[0] == 'left':
                    times_won += 1

    print("-> Cheat sheet generated.")
    return times_won / rollouts


def generate_utility_matrix(cards_on_table: [], full_deck: bool):
    # Create a fake deck.
    deck = np.array(correct_format(create_deck(full_deck)))
    # Get all hole pairs.
    cards_deck_combos = [[t1, t2] for i, t1 in enumerate(deck) for t2 in deck[i + 1:]]
    # Generate utility matrix.
    hole_card_combinations = len(cards_deck_combos)
    utility_matrix = np.zeros((hole_card_combinations, hole_card_combinations))

    # Find the 1, -1 and 0s for the utility matrix.
    for i, hole_cards1 in enumerate(cards_deck_combos):
        for j, hole_cards2 in enumerate(cards_deck_combos):
            if set(hole_cards1) & set(hole_cards2) or set(hole_cards1) & set(cards_on_table) or set(hole_cards2) & set(
                    cards_on_table):
                # Impossible scenario, since there cannot be duplicate cards
                utility_matrix[i, j] = 0
            else:
                if len(cards_on_table) == 0:
                    hand1 = list(hole_cards1)
                    hand2 = list(hole_cards2)
                    winner = compare_hands_2cards(hand1, hand2)
                else:
                    hand1 = list(hole_cards1 + cards_on_table)
                    hand2 = list(hole_cards2 + cards_on_table)
                    winner = compare_hands(hand1, hand2)
                utility_matrix[i, j] = 1 if winner[0] == "left" else -1

    # Draw scenarios, where both hole card combinations are the same
    np.fill_diagonal(utility_matrix, 0)
    return utility_matrix


# gets the most common element from a list
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0]


def correct_format(cards: []):
    a = []
    for card in cards:
        a.append(str(card.value) + card.color[0].upper())
    return a


# gets card value from  a hand. converts A to 14,  is_seq function will convert the 14 to a 1 when necessary to evaluate A 2 3 4 5 straights
def convert_tonums(h, nums = {'T':10, 'J':11, 'Q':12, 'K':13, "A": 14}):
    for x in range(len(h)):

        if (h[x][0]) in nums.keys():

            h[x] = str(nums[h[x][0]]) + h[x][1]

    return h


# is royal flush
# if a hand is a straight and a flush and the lowest value is a 10 then it is a royal flush
def is_royal(h):
    nh = convert_tonums(h)
    if is_seq(h):
        if is_flush(h):
            nn = [int(x[:-1]) for x in nh]
            if min(nn) == 10:
                return True

    else:
        return False


# converts hand to number valeus and then evaluates if they are sequential  AKA a straight
def is_seq(h):
    ace = False
    r = h[:]

    h = [x[:-1] for x in convert_tonums(h)]


    h = [int(x) for x in h]
    h = list(sorted(h))
    ref = True
    for x in range(0,len(h)-1):
        if not h[x]+1 == h[x+1]:
            ref =  False
            break

    if ref:
        return True, r

    aces = [i for i in h if str(i) == "14"]
    if len(aces) == 1:
        for x in range(len(h)):
            if str(h[x]) == "14":
                h[x] = 1

    h = list(sorted(h))
    for x in range(0,len(h)-1):
        if not h[x]+1 == h[x+1]:

            return False
    return True, r

# call set() on the suite values of the hand and if it is 1 then they are all the same suit
def is_flush(h):
    suits = [x[-1] for x in h]
    if len(set(suits)) == 1:
        return True, h
    else:
        return False


# if the most common element occurs 4 times then it is a four of a kind
def is_fourofakind(h):
    h = [a[:-1] for a in h]
    i = Most_Common(h)
    if i[1] == 4:
        return True, i[0]
    else:
        return False


# if the most common element occurs 3 times then it is a three of a kind
def is_threeofakind(h):
    h = [a[:-1] for a in h]
    i = Most_Common(h)
    if i[1] == 3:
        return True, i[0]
    else:
        return False


# if the first 2 most common elements have counts of 3 and 2, then it is a full house
def is_fullhouse(h):
    h = [a[:-1] for a in h]
    data = Counter(h)
    a, b = data.most_common(1)[0], data.most_common(2)[-1]
    if str(a[1]) == '3' and str(b[1]) == '2':
        return True, (a, b)
    return False


# if the first 2 most common elements have counts of 2 and 2 then it is a two pair
def is_twopair(h):
    h = [a[:-1] for a in h]
    data = Counter(h)
    a, b = data.most_common(1)[0], data.most_common(2)[-1]
    if str(a[1]) == '2' and str(b[1]) == '2':
        return True, (a[0], b[0])
    return False


#if the first most common element is 2 then it is a pair
# DISCLAIMER: this will return true if the hand is a two pair, but this should not be a conflict because is_twopair is always evaluated and returned first
def is_pair(h):
    h = [a[:-1] for a in h]
    data = Counter(h)
    a = data.most_common(1)[0]

    if str(a[1]) == '2':
        return True, (a[0])
    else:
        return False


# get the high card
def get_high(h):
    return list(sorted([int(x[:-1]) for x in convert_tonums(h)], reverse =True))[0]


# FOR HIGH CARD or ties, this function compares two hands by ordering the hands from highest to lowest and comparing each card and returning when one is higher then the other
def compare(xs, ys):
  xs, ys = list(sorted(xs, reverse =True)), list(sorted(ys, reverse = True))

  for i, c in enumerate(xs):
    if ys[i] > c:
        return 'RIGHT'
    elif ys[i] < c:
        return 'LEFT'

    return "TIE"


# categorized a hand based on previous functions
def evaluate_hand(h):

    if is_royal(h):
        return "ROYAL FLUSH", h, 10
    elif is_seq(h) and is_flush(h) :
        return "STRAIGHT FLUSH", h, 9
    elif is_fourofakind(h):
        _, fourofakind = is_fourofakind(h)
        return "FOUR OF A KIND", fourofakind, 8
    elif is_fullhouse(h):
        return "FULL HOUSE", h, 7
    elif is_flush(h):
        _, flush = is_flush(h)
        return "FLUSH", h, 6
    elif is_seq(h):
        _, seq = is_seq(h)
        return "STRAIGHT", h, 5
    elif is_threeofakind(h):
        _, threeofakind = is_threeofakind(h)
        return "THREE OF A KIND", threeofakind, 4
    elif is_twopair(h):
        _, two_pair = is_twopair(h)
        return "TWO PAIR", two_pair, 3
    elif is_pair(h):
        _, pair = is_pair(h)
        return "PAIR", pair, 2
    else:
        return "HIGH CARD", h, 1


# this monster function evaluates two hands and also deals with ties and edge cases
# this probably should be broken up into separate functions but aint no body got time for that
def compare_hands(h1,h2):
    one, two = evaluate_hand(h1), evaluate_hand(h2)
    if one[0] == two[0]:

        if one[0] =="STRAIGHT FLUSH":

            sett1, sett2 = convert_tonums(h1), convert_tonums(h2)
            sett1, sett2 = [int(x[:-1]) for x in sett1], [int(x[:-1]) for x in sett2]
            com = compare(sett1, sett2)

            if com == "TIE":
                return "none", one[1], two[1]
            elif com == "RIGHT":
                return "right", two[0], two[1]
            else:
                return "left", one[0], one[1]

        elif one[0] == "TWO PAIR":

            leftover1, leftover2 = is_twopair(h1), is_twopair(h2)
            twm1, twm2 = max([int(x) for x in list(leftover1[1])]), max([int(x) for x in list(leftover2[1])])
            if twm1 > twm2:
                return "left", one[0], one[1]
            elif twm1 < twm2:
                return "right", two[0], two[1]


            if compare(list(leftover1[1]), list(leftover2[1])) == "TIE":
                l1 = [x[:-1] for x in h1 if x[:-1] not in leftover1[1]]
                l2 = [x[:-1] for x in h2 if x[:-1] not in leftover2[1]]
                if int(l1[0]) == int(l2[0]):
                    return "none", one[1], two[1]
                elif int(l1[0]) > int(l2[0]):
                    return "left", one[0], one[1]
                else:
                    return "right", two[0], two[1]
            elif compare(list(leftover1[1]), list(leftover2[1]))  == "RIGHT":
                return "right", two[0], two[1]
            elif  compare(list(leftover1[1]), list(leftover2[1]))  == "LEFT":
                return "left", one[0], one[1]


        elif one[0] == "PAIR":
            sh1, sh2 = int(is_pair(h1)[1]), int(is_pair(h2)[1])
            if sh1 == sh2:

                c1 = [int(x[:-1]) for x in convert_tonums(h1) if not int(sh1) == int(x[:-1])]
                c2 = [int(x[:-1]) for x in convert_tonums(h2) if not int(sh1) == int(x[:-1])]
                if compare(c1, c2) == "TIE":
                    return "none", one[1], two[1]
                elif compare(c1, c2) == "RIGHT":
                    return "right", two[0], two[1]
                else:
                    return "left", one[0], one[1]




            elif h1 > h2:
                return "right", two[0], two[1]
            else:
                return "left", one[0], one[1]

        elif one[0] == 'FULL HOUSE':

            fh1, fh2 =  int(is_fullhouse(h1)[1][0][0]), int(is_fullhouse(h2)[1][0][0])
            if fh1 > fh2:
                return "left", one[0], one[1]
            else:
                return "right", two[0], two[1]
        elif one[0] == "HIGH CARD":
            sett1, sett2 = convert_tonums(h1), convert_tonums(h2)
            sett1, sett2 = [int(x[:-1]) for x in sett1], [int(x[:-1]) for x in sett2]
            com = compare(sett1, sett2)
            if com == "TIE":
                return "none", one[1], two[1]
            elif com == "RIGHT":
                return "right", two[0], two[1]
            else:
                return "left", one[0], one[1]

        elif len(one[1]) < 5:
            if max(one[1])  == max(two[1]):
                return "none", one[1], two[1]
            elif max(one[1]) > max(two[1]):
                return "left", one[0], one[1]
            else:
                return "right", two[0], two[1]
        else:
            n_one, n_two = convert_tonums(one[1]), convert_tonums(two[1])
            n_one, n_two = [int(x[:-1]) for x in n_one], [int(x[:-1]) for x in n_two]

            if max(n_one)  == max(n_two):
                return "none", one[1], two[1]
            elif max(n_one) > max(n_two):
                return "left", one[0], one[1]
            else:
                return "right", two[0], two[1]
    elif one[2] > two[2]:
        return "left", one[0], one[1]
    else:
        return "right", two[0], two[1]


# Function to compare 2-card hands
def compare_hands_2cards(h1, h2):
    h1_high = get_high(h1)
    h2_high = get_high(h2)
    if h1_high > h2_high:
        return "left", "HIGH CARD", h1
    elif h1_high < h2_high:
        return "right", "HIGH CARD", h2
    else:
        return "none", "HIGH CARD", h1, h2


def get_ai_names():
    names = ["Bot Steve", "Bot Butch", "Bot Frank", "Bot Terminator",
             "Bot Cheater", "Bot God", "Bot Hacker", "Bot Michael"]
    return names


def create_deck(full_deck: bool):
    # Assign the minimum card in the deck. Default 2.
    min_card = 2
    if not full_deck:
        min_card = 9

    # The available colors.
    colors = ['heart', 'diamonds', 'spades', 'clubs']

    # Create the deck.
    deck = [Card(value, color) for value in range(min_card, 15) for color in colors]

    return deck


def shuffle_deck(deck: [], count):
    # The number of shuffles can't be negative.
    if count < 0:
        print("ERROR: The number of deck shuffles has be 0 or greater.")
        return

    # Shuffle the specified number of times.
    shuffles = 0
    while shuffles < count:
        # Move every card to a random position in the deck.
        for old_position in range(len(deck)):
            new_position = random.randint(0, len(deck))
            deck.insert(new_position, deck.pop(old_position))

        shuffles += 1

    return deck


def show_deck(deck: []):
    string = ""
    for card in deck:
        value = card.value
        if card.value == 11:
            value = "J"
        elif card.value == 12:
            value = "Q"
        elif card.value == 13:
            value = "K"
        elif card.value == 14:
            value = "A"
        string += str(str(value) + str(card.color[0].upper()) + " ")
    print(string)


