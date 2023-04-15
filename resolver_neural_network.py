import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from oracle import create_deck, shuffle_deck, generate_utility_matrix, correct_format
from helper_functions import card_to_index, combination_idx_to_card_pair, card_str_to_color_value


def one_hot_encode_cards(cards_on_table, full_deck):
    num_cards_on_table = len(cards_on_table)
    deck_size = 52 if full_deck else 24
    one_hot_size = deck_size

    one_hot_encoded = np.zeros((num_cards_on_table, one_hot_size), dtype=int)

    for i, card_str in enumerate(cards_on_table):
        card_obj = card_str_to_color_value(card_str)
        card_idx = card_to_index(card_obj, full_deck)  # Convert card string to index

        # Skip the index if it's out of bounds for the one_hot_size.
        if card_idx < one_hot_size:
            one_hot_encoded[i, card_idx] = 1

    return one_hot_encoded


def generate_neural_network(input_dim, hidden_layers, output_dim):
    # Input layers
    if full_deck:
        num_cards = 52
    else:
        num_cards = 24
    p1_ranges = Input(shape=(input_dim[0],), name="p1_ranges")
    p2_ranges = Input(shape=(input_dim[1],), name="p2_ranges")
    cards_on_table = Input(shape=(max_cards_on_table, num_cards), name="cards_on_table")
    pot_size = Input(shape=(input_dim[2],), name="pot_size")

    # Process cards_on_table with GlobalAveragePooling1D
    cards_processed = tf.keras.layers.GlobalAveragePooling1D()(cards_on_table)

    # Concatenate inputs
    x = Concatenate()([p1_ranges, p2_ranges, cards_processed, pot_size])

    # Hidden layers
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)

    # Output layer
    output = Dense(output_dim, activation='softmax')(x)

    # Create and return model.
    return Model(inputs=[p1_ranges, p2_ranges, cards_on_table, pot_size], outputs=output)


def generate_training_data(num_cards_on_table: int, full_deck: bool, rollouts):
    if full_deck:
        num_cards = 52
    else:
        num_cards = 24

    # Go through each rollout.
    x1 = np.zeros((rollouts, int((num_cards*(num_cards-1))/2)))
    x2 = np.zeros((rollouts, int((num_cards*(num_cards-1))/2)))
    x3 = np.zeros((rollouts, num_cards_on_table, num_cards))
    x4 = np.zeros((rollouts, 1))

    y = np.zeros((rollouts, (int((num_cards*(num_cards-1))/2))*2))

    for i in range(rollouts):
        # Ranges. P1 & P2 (random initially)
        if full_deck:
            p1_range = np.array([random.randint(0, 100) for i in range(1326)])
            p2_range = np.array([random.randint(0, 100) for i in range(1326)])

            list_of_pairs = [combination_idx_to_card_pair(i, full_deck) for i in range(1326)]
        else:
            p1_range = np.array([random.randint(0, 100) for i in range(276)])
            p2_range = np.array([random.randint(0, 100) for i in range(276)])

            list_of_pairs = [combination_idx_to_card_pair(i, full_deck) for i in range(276)]

        # Pot.
        pot_size_min = 15
        pot_size_max = 200
        pot_size = random.randint(pot_size_min, pot_size_max)

        # Generate cards on table. 5 for river, etc.
        deck = shuffle_deck(create_deck(full_deck), 69)
        cards_on_table = []
        for i in range(num_cards_on_table):
            card = deck.pop()
            cards_on_table.append(card)

        # Remove the cards on the table from the player ranges.
        for card in cards_on_table:
            all_indices_with_card = [
                i for i, pair in enumerate(list_of_pairs) if card in pair
            ]
            p1_range[all_indices_with_card] = 0
            p2_range[all_indices_with_card] = 0

        cards_on_table = correct_format(cards_on_table)
        M = generate_utility_matrix(cards_on_table, full_deck)

        v1 = M.dot(p2_range.transpose())
        v2 = ((-1)*p1_range).dot(M)

        # one-hot-encode cards
        cards_on_table = one_hot_encode_cards(cards_on_table, full_deck)
        x1[i] = np.array(p1_range)/np.sum(p1_range)
        x2[i] = np.array(p2_range)/np.sum(p2_range)
        x3[i] = np.array(cards_on_table)
        x4[i] = np.array(pot_size / pot_size_max)

        y[i] = np.concatenate([np.array(v1), np.array(v2)])

    return [x1, x2, x3, x4], y


full_deck = False

if full_deck:
    num_cards = 52
else:
    num_cards = 24
max_cards_on_table = 5  # The maximum number of cards on the table (e.g., the river)

input_dim = (int((num_cards*(num_cards-1))/2), int((num_cards*(num_cards-1))/2), 1)  # P1 ranges, P2 ranges, pot size, (public cards)
hidden_layers = [256, 128, 64]
output_dim = int((num_cards*(num_cards-1))/2) + int((num_cards*(num_cards-1))/2)  # Fold, Call, Raise
model = generate_neural_network(input_dim, hidden_layers, output_dim)
rollouts = 5

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# X_train is a list of 4 numpy arrays: p1_ranges, p2_ranges, cards_on_table, pot_size
# y_train is a numpy array of one-hot encoded actions (Fold, Call, Raise)
X_train, y_train = generate_training_data(5, full_deck, rollouts)  # River

# Fit the model.
model.fit(X_train, y_train, batch_size=1, epochs=25, validation_split=0.2)

