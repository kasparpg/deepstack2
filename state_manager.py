

class GameState:
    def __init__(self, players: [], my_index: int, cards_on_table: [], highest_bid: int, deck: [], lap: int, fake_state: bool):
        self.fake_state = fake_state

        # Table states.
        self.cards_on_table = cards_on_table
        self.chips_on_table = 0
        self.highest_bid = highest_bid
        self.num_players = len(players)
        self.deck = deck
        self.lap = lap  # lap = 0 means pre-flop, lap = 1 means flop etc.
        self.bet_limit = 10

        # My states.
        self.name = players[my_index].name
        self.my_index = my_index
        self.my_cards = []
        self.my_chips = 0
        self.my_chips_on_table = 0
        self.taken_action = False

        # Other player states.
        self.players = players
        self.folded_players = []
        self.player_chips = []
        self.player_chips_on_table = []

        for player in players:
            # Keep track of my own cards.
            if players.index(player) == self.my_index:
                self.my_cards = player.cards
                self.my_chips = player.chips
                self.my_chips_on_table = player.chips_added_to_table

            # Keep track of players who have folded.
            if player.folded:
                self.folded_players.append(True)
            else:
                self.folded_players.append(False)

            # Keep track of player chips and chips added to table.
            self.player_chips.append(player.chips)
            self.player_chips_on_table.append(player.chips_added_to_table)
            self.chips_on_table += player.chips_added_to_table


def check_if_all_players_taken_action(players, highest_bid: int):
    all_players_taken_action = True
    for player in players:
        if player.folded:
            continue
        if not player.action_taken or player.chips_added_to_table != highest_bid:
            all_players_taken_action = False
    return all_players_taken_action


def determine_next_state(game_state: GameState):
    # If the state is fake, we can modify it to build a tree.
    if game_state.fake_state:
        if check_if_all_players_taken_action(game_state.players, game_state.highest_bid):
            game_state.lap += 1
            for player in game_state.players:
                player.action_taken = False
        if game_state.lap == 0:
            pass
        elif game_state.lap == 1:
            while len(game_state.cards_on_table) < 3:
                card = game_state.deck.pop()
                game_state.cards_on_table.append(card)
        # The second betting round is over. Add another card to the table.
        elif game_state.lap == 2 or game_state.lap == 3:
            card = game_state.deck.pop()
            game_state.cards_on_table.append(card)

    return game_state
