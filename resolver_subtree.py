import itertools
from copy import deepcopy
from helper_functions import get_available_actions
from resolver import take_action
from state_manager import GameState
import numpy as np
import random


class Node:
    def __init__(self, player, actions, parent=None, payoffs=None, terminal=False):
        p1_range = np.array([random.randint(0, 100) for i in range(1326)])
        self.p1_range = np.array(p1_range)/np.sum(p1_range)
        p2_range = np.array([random.randint(0, 100) for i in range(1326)])
        self.p2_range = np.array(p2_range)/np.sum(p2_range)

        self.player = player
        self.actions = actions
        # self.strategy = range*actions
        self.parent = parent
        self.children = []
        self.payoffs = payoffs
        self.terminal = terminal


def build_subtree(game_state, depth=0, max_depth=10):
    # print("> Subtree generated with depth " + str(depth) + ".")
    game_state = GameState(
        players=deepcopy(game_state.players),
        my_index=game_state.my_index,
        cards_on_table=deepcopy(game_state.cards_on_table),
        highest_bid=game_state.highest_bid,
        deck=deepcopy(game_state.deck),
        lap=game_state.lap,
        fake_state=True)

    if depth >= max_depth or is_terminal(game_state):
        # print(">>> Terminal node reached. (depth = " + str(depth) + ", lap = " + str(game_state.lap)+")")
        payoffs = compute_payoffs(game_state) if is_terminal(game_state) else None
        terminal_node = Node(game_state.players[game_state.my_index], [], terminal=True, payoffs=payoffs)
        terminal_node.children = []
        return terminal_node

    node = Node(game_state.players[game_state.my_index], [])
    actions = get_available_actions(game_state)

    for action in actions:
        # print(">> Action is "+str(action) + ". Creating child node...")
        next_game_state = take_action(game_state, action)
        child = build_subtree(next_game_state, depth + 1, max_depth)
        node.children.append(child)
        node.actions.append(action)
    return node


def is_terminal(game_state):
    if game_state.lap == 4:
        return True
    if game_state.folded_players[0] or game_state.folded_players[1]:
        return True
    return False


def compute_payoffs(game_state):

    return 0

