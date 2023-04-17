import itertools
from copy import deepcopy
from helper_functions import get_available_actions, combination_idx_to_card_pair
from resolver import take_action
from state_manager import GameState
import numpy as np
import random

full_deck = False
pot_size_max = 200


class Node:
    def __init__(self, player, actions, parent=None, payoffs=None, terminal=False, cards_on_table=[], pot_size=0):
        self.cards_on_table = cards_on_table
        self.pot_size = pot_size
        list_of_pairs = [combination_idx_to_card_pair(i, full_deck) for i in range(276)]
        if not parent:
            p1_range = np.array([random.randint(0, 100) for i in range(276)])
            self.p1_range = np.array(p1_range)/np.sum(p1_range)

            p2_range = np.array([random.randint(0, 100) for i in range(276)])
            self.p2_range = np.array(p2_range)/np.sum(p2_range)
        else:
            self.p1_range = np.ones(276)
            self.p2_range = np.ones(276)

        # Set cards on table to 0 in range.
        for card in self.cards_on_table:
            all_indices_with_card = [
                i for i, pair in enumerate(list_of_pairs) if card in pair
            ]
            self.p1_range[all_indices_with_card] = 0
            self.p2_range[all_indices_with_card] = 0

        self.player = player
        self.actions = actions

        self.strategy_array = None

        self.v1_range = None
        self.v2_range = None

        self.parent = parent
        self.children = []
        self.payoffs = payoffs
        self.terminal = terminal

    def initialize_strategy_array(self, actions_length):
        strategy_array = np.array([random.uniform(0, 1) for i in range(276 * actions_length)])\
            .reshape(276, actions_length)
        self.strategy_array = strategy_array / strategy_array.sum(axis=1)[:, None]


def build_subtree(game_state, depth=0, max_depth=5, parent=None):
    game_state = GameState(
        players=deepcopy(game_state.players),
        my_index=game_state.my_index,
        cards_on_table=deepcopy(game_state.cards_on_table),
        highest_bid=game_state.highest_bid,
        deck=deepcopy(game_state.deck),
        lap=game_state.lap,
        fake_state=True)

    if depth >= max_depth or is_terminal(game_state):
        payoffs = compute_payoffs(game_state) if is_terminal(game_state) else None
        terminal_node = Node(game_state.players[game_state.my_index], [], parent, terminal=True, payoffs=payoffs, cards_on_table=game_state.cards_on_table, pot_size=game_state.chips_on_table)
        terminal_node.children = []
        return terminal_node

    node = Node(game_state.players[game_state.my_index], [], parent, game_state.cards_on_table)
    actions = get_available_actions(game_state)
    node.initialize_strategy_array(len(actions))
    for i, action in enumerate(actions):
        a_i = i  # action index
        p_a = np.sum(node.strategy_array[a_i], axis=0) / np.sum(np.sum(node.strategy_array[a_i], axis=0), axis=0)  # prob(action)

        next_game_state = take_action(game_state, action)
        child = build_subtree(next_game_state, depth + 1, max_depth, node)

        if child.player.human:
            for i, hole in enumerate(child.p1_range):
                child.p1_range[i] *= (node.strategy_array[i][a_i]*node.p1_range[i]) / p_a
            if child.terminal:
                if action == 'FOLD':
                    child.v1_range = np.zeros(276)
                    child.v2_range = np.zeros(276)
                    for i, hole in enumerate(child.p1_range):
                        if hole > 0:
                            child.v1_range[i] = child.pot_size / pot_size_max
                            child.v2_range[i] = - (child.pot_size / pot_size_max)

        else:
            for i, hole in enumerate(child.p2_range):
                child.p2_range[i] *= (node.strategy_array[i][a_i] * node.p2_range[i]) / p_a
            if child.terminal:
                if action == 'FOLD':
                    child.v1_range = np.zeros(276)
                    child.v2_range = np.zeros(276)
                    for i, hole in enumerate(child.p2_range):
                        if hole > 0:
                            child.v1_range[i] = - (child.pot_size / pot_size_max)
                            child.v2_range[i] = child.pot_size / pot_size_max

        node.children.append(child)
        node.actions.append(action)
    return node


def update_tree(node, depth=0, action=None):
    # Ranges
    for action, child in zip(node.actions, node.children):
        a_i = node.actions.index(action)  # action index
        p_a = np.sum(node.strategy_array[a_i], axis=0) / np.sum(np.sum(node.strategy_array[a_i], axis=0),
                                                                axis=0)  # prob(action)
        if child.player.human:
            for i, hole in enumerate(child.p1_range):
                child.p1_range[i] *= (node.strategy_array[i][a_i]*node.p1_range[i]) / p_a
            if child.terminal:
                if action == 'FOLD':
                    child.v1_range = np.zeros(276)
                    child.v2_range = np.zeros(276)
                    for i, hole in enumerate(child.p1_range):
                        if hole > 0:
                            child.v1_range[i] = child.pot_size / pot_size_max
                            child.v2_range[i] = - (child.pot_size / pot_size_max)
        else:
            for i, hole in enumerate(child.p2_range):
                child.p2_range[i] *= (node.strategy_array[i][a_i] * node.p2_range[i]) / p_a
            if child.terminal:
                if action == 'FOLD':
                    child.v1_range = np.zeros(276)
                    child.v2_range = np.zeros(276)
                    for i, hole in enumerate(child.p2_range):
                        if hole > 0:
                            child.v1_range[i] = - (child.pot_size / pot_size_max)
                            child.v2_range[i] = child.pot_size / pot_size_max

        update_tree(child, depth=depth + 1, action=action)


def is_terminal(game_state):
    if game_state.lap == 4:
        return True
    if game_state.folded_players[0] or game_state.folded_players[1]:
        return True
    return False



def compute_payoffs(game_state):

    return 0

