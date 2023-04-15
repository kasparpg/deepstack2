import numpy as np
from state_manager import GameState


# Make sure the input is an integer
def input_number(message):
    while True:
        try:
            user_input = int(input(message))
        except ValueError:
            print("ERROR: Not an integer! Try again.")
            continue
        else:
            return user_input


# Make sure array indices are correct, by looping back to the start of the array when going above the highest index.
def get_proper_array_index(index, array, increment):
    return (index+increment) % len(array)


# Check if every player has taken an action.
def check_if_all_players_taken_action(players, highest_bid: int):
    all_players_taken_action = True
    for player in players:
        if player.folded:
            continue
        if not player.action_taken or player.chips_added_to_table != highest_bid:
            all_players_taken_action = False
    return all_players_taken_action


# Show leaderboard.
def leaderboard(players: []):
    chips = []
    names = []
    for player in players:
        chips.append(player.chips)
        names.append(player.name)

    names = [x for y, x in sorted(zip(chips, names))]
    chips.sort(reverse=True)
    i = 1
    print("\nThe standings are:")
    for name, chip in zip(reversed(names), chips):
        print(str(i)+".", name, "has", chip, "chips.")
        i += 1


# Check if a bid is the highest bid.
def check_highest_bid(bid: int, highest_bid: int, fake_state: bool):
    if bid > highest_bid:
        if not fake_state:
            print("The highest bid is now " + str(bid) + " chips.")
        return bid
    return highest_bid


# Make sure the action the player is taking is valid.
def check_legal_action(action: int, game_state: GameState, chips_to_give: int):
    action_index = game_state.my_index
    current_player = game_state.players[action_index]
    highest_bid = game_state.highest_bid
    table_chips = game_state.chips_on_table
    cards_on_table = game_state.cards_on_table
    players = game_state.players
    fake_state = game_state.fake_state

    if 0 < action < 6:
        # Raise.
        if action == 1:
            if current_player.human:
                if fake_state:
                    chips_to_give = highest_bid-current_player.chips_added_to_table
                else:
                    chips_to_give = input_number("\nHow many chips would you like to add to the table? (" + str(
                    highest_bid - current_player.chips_added_to_table) + " to call). Enter here: ")
            else:
                chips_to_give = chips_to_give
            if current_player.action_taken and chips_to_give != (highest_bid - current_player.chips_added_to_table):
                print("You have already taken an action. Call or fold.")
            # Check if the player has enough chips.
            elif chips_to_give > current_player.chips:
                # If the player has fewer chips to call, create a sidepot.
                if current_player.chips < (highest_bid - current_player.chips_added_to_table):
                    # side-pot
                    print("\nside-pot stuff??")
                # The player has enough chips to call, but entered more than he has.
                else:
                    print("\nYou don't have that many chips.")
            # The player must add at least 0 chips. 0 to add none, and skip their turn.
            elif chips_to_give < 0:
                print("You must add at least 1 chip.")
            # The player must add at least the amount to call.
            elif (chips_to_give + current_player.chips_added_to_table) < highest_bid:
                print("You have to add at least", (highest_bid -
                                                   current_player.chips_added_to_table),
                      "chips to the table. Did you mean to fold? Press 2.")
            # The player has correctly added enough chips.
            else:
                highest_bid = check_highest_bid(chips_to_give + current_player.chips_added_to_table, highest_bid, fake_state)
                table_chips += chips_to_give
                if not fake_state:
                    print("->", current_player.name, "has added", chips_to_give, "to the table.")
                current_player.chips -= chips_to_give
                current_player.chips_added_to_table += chips_to_give
                current_player.action_taken = True
                return action_index, highest_bid, chips_to_give
        # Fold.
        elif action == 2:
            if not fake_state:
                print("\n->", current_player.name, "has folded.")
            current_player.folded = True
            return action_index, check_highest_bid(0, highest_bid, fake_state), 0
        # See your own cards.
        elif action == 3:
            cards = ""
            for card in current_player.cards:
                cards += str(card.value)
                cards += str(card.color)
                cards += " "
            print("\nYour cards are:", cards)
        # See the cards on the table.
        elif action == 4:
            if len(cards_on_table) > 0:
                cards = ""
                for card in cards_on_table:
                    cards += str(card.value)
                    cards += str(card.color)
                    cards += " "
                print("\nThe cards on the table are:", cards)
            else:
                print("\nThere are no cards on the table yet.")
        # Show the player standings.
        else:
            leaderboard(players)
        return -1, highest_bid, 0


# Get the actions a bot can take.
def get_available_actions(game_state: GameState):
    available_actions = ["FOLD"]
    # Check if the bot has more or equal chips than the highest bid. (To call or raise)
    if game_state.my_chips >= game_state.highest_bid - game_state.my_chips_on_table:
        available_actions.append("CALL")
        # Check if the bot has more chips than the highest bid. (To raise)
        if game_state.my_chips > game_state.highest_bid - game_state.my_chips_on_table and not game_state.taken_action:
            available_actions.append("RAISE")
    return available_actions


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        print(a)
        arr[...,i] = a
    return arr.reshape(-1, la)


def display_tree(node, depth=0, action=None):
    indent = "  " * depth
    if action is not None:
        action_str = f"Action: {action}, "
    else:
        action_str = ""

    if node.terminal:
        payoffs_str = f", Payoffs: {node.payoffs}"
    else:
        payoffs_str = ""

    print(f"{indent}{action_str}Player: {node.player.name}{payoffs_str}")

    for action, child in zip(node.actions, node.children):
        display_tree(child, depth + 1, action)


def card_to_index(card, full_deck=True):
    """
    Returns a unique number for every card,
    given the card's suit and rank.
    """
    suit_values = {"heart": 0, "spades": 12 + 1, "diamond": 12 * 2 + 1, "clubs": 12 * 3 + 1}
    card_index = suit_values[card['color']] + card['value']

    if not full_deck:
        # Adjust the card index to match the reduced deck size.
        card_index = card_index % 24

    return card_index



def combination_idx_to_card_pair(combination_idx, full_deck):
    """
    Combination idx is a number between 0 and 1325. This function returns
    a pair of cards of the type (card_1, card_2),
    for example (2, 51)

    2 -> (0, 1)
    10 -> (0, 11)

    125 -> (12, 53)
    """
    if full_deck:
        all_unique_pairs = list(
            set([(i, j) for i in range(52) for j in range(51)]))  # 1326 elements
    else:
        all_unique_pairs = list(
            set([(i, j) for i in range(24) for j in range(23)]))  # 276 elements

    card_pair = all_unique_pairs[combination_idx]

    return card_pair


def card_str_to_color_value(card_str):
    value_str = card_str[:-1]
    color_str = card_str[-1]

    color_dict = {"h": "heart", "s": "spades", "d": "diamond", "c": "clubs"}

    value_dict = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "11": 11, "12": 12, "13": 13, "14": 14
    }

    color = color_dict.get(color_str.lower())
    value = value_dict.get(value_str)

    return {"color": color, "value": value}
