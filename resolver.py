from state_manager import GameState, determine_next_state
from helper_functions import input_number, get_proper_array_index
import random
from helper_functions import check_legal_action
import tensorflow as tf
import numpy as np
from keras import layers
from copy import deepcopy


def take_action(game_state: GameState, action: str):
    if game_state.fake_state:
        game_state = GameState(
            players=deepcopy(game_state.players),
            my_index=game_state.my_index,
            cards_on_table=deepcopy(game_state.cards_on_table),
            highest_bid=game_state.highest_bid,
            deck=deepcopy(game_state.deck),
            lap=game_state.lap,
            fake_state=True)

    # The bot folds.
    if action == "FOLD":
        action_index, highest_bid, chips_to_give = check_legal_action(2, game_state, 0)

    # The bot calls.
    elif action == "CALL":
        chips_to_give = game_state.highest_bid-game_state.my_chips_on_table
        action_index, highest_bid, chips_to_give = check_legal_action(1, game_state, chips_to_give)
        game_state.my_chips_on_table += chips_to_give
    # The bot raises.
    else:
        chips_to_give = random.randint(game_state.highest_bid - game_state.my_chips_on_table, game_state.my_chips)
        action_index, highest_bid, chips_to_give = check_legal_action(1, game_state, chips_to_give)
        game_state.my_chips_on_table += chips_to_give
    return determine_next_state(GameState(
        players=game_state.players,
        my_index=get_proper_array_index(action_index, game_state.players, 1),
        cards_on_table=game_state.cards_on_table,
        highest_bid=highest_bid,
        deck=game_state.deck,
        lap=game_state.lap,
        fake_state=game_state.fake_state))

