import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import numpy as np
import tensorflow as tf
import pickle
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from oracle import create_deck, shuffle_deck, generate_utility_matrix, correct_format
from helper_functions import card_to_index, combination_idx_to_card_pair, card_str_to_color_value


def one_hot_encode_cards(cards_on_table, full_deck):
    num_cards_on_table = len(cards_on_table)
    deck_size = 52 if full_deck else 24
    one_hot_size = deck_size

    one_hot_encoded = np.zeros((num_cards_on_table, one_hot_size), dtype=int)

    for i, card in enumerate(cards_on_table):
        card_idx = card_to_index(card, full_deck)  # Convert card string to index

        # Skip the index if it's out of bounds for the one_hot_size.
        if card_idx < one_hot_size:
            one_hot_encoded[i, card_idx] = 1

    return one_hot_encoded


def generate_neural_network(input_dim, hidden_layers, output_dim, num_cards_on_table):
    # Input layers
    if full_deck:
        num_cards = 52
    else:
        num_cards = 24
    p1_ranges = Input(shape=(input_dim[0],), name="p1_ranges")
    p2_ranges = Input(shape=(input_dim[1],), name="p2_ranges")
    cards_on_table = Input(shape=(num_cards_on_table, num_cards), name="cards_on_table")
    pot_size = Input(shape=(input_dim[2],), name="pot_size")

    # Process cards_on_table with GlobalAveragePooling1D
    cards_processed = tf.keras.layers.GlobalAveragePooling1D()(cards_on_table)

    # Concatenate inputs
    x = Concatenate()([p1_ranges, p2_ranges, cards_processed, pot_size])

    # Hidden layers
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)

    # Output layer
    output = Dense(output_dim)(x)

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

    M = 0
    for i in range(rollouts):
        print("-> Rollout", i+1, "of", rollouts, "rollouts.")
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
        for j in range(num_cards_on_table):
            card = deck.pop()
            cards_on_table.append(card)

        # Remove the cards on the table from the player ranges.
        for card in cards_on_table:
            c = card_to_index(card, full_deck)
            all_indices_with_card = [
                i for i, pair in enumerate(list_of_pairs) if c in pair
            ]
            p1_range[all_indices_with_card] = 0
            p2_range[all_indices_with_card] = 0

        one_hot_cards = one_hot_encode_cards(cards_on_table, full_deck)
        if i == 0 or i % 25 == 0:
            cards_on_table = correct_format(cards_on_table)
            M = generate_utility_matrix(cards_on_table, full_deck)

        v1 = M.dot(p2_range.transpose())
        v2 = ((-1)*p1_range).dot(M)

        x1[i] = np.array(p1_range)/np.sum(p1_range)
        x2[i] = np.array(p2_range)/np.sum(p2_range)
        x3[i] = np.array(one_hot_cards)  # one-hot-encode cards
        x4[i] = np.array(pot_size / pot_size_max)

        y[i] = np.concatenate([np.array(v1), np.array(v2)])
    return [x1, x2, x3, x4], y


full_deck = False

if full_deck:
    num_cards = 52
else:
    num_cards = 24  # The maximum number of cards on the table (e.g., the river)

num_cards_on_table = 5
rollouts = 10000

input_dim = (int((num_cards*(num_cards-1))/2), int((num_cards*(num_cards-1))/2), 1)  # P1 ranges, P2 ranges, pot size, (public cards)
hidden_layers = [256, 128, 64]
output_dim = int((num_cards*(num_cards-1))/2) + int((num_cards*(num_cards-1))/2)  # Fold, Call, Raise
model = generate_neural_network(input_dim, hidden_layers, output_dim, num_cards_on_table)

model.compile(optimizer='adam', loss='MSE')

# X_train is a list of 4 numpy arrays: p1_ranges, p2_ranges, cards_on_table, pot_size
# y_train is a numpy array of one-hot encoded actions (Fold, Call, Raise)
X_train, y_train = generate_training_data(num_cards_on_table, full_deck, rollouts)  # River


with open('data/x_train'+str(num_cards)+'_'+str(num_cards_on_table)+'cards_'+str(rollouts) + 'rollouts.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(X_train, file)

with open('data/y_train'+str(num_cards)+'_'+str(num_cards_on_table)+'cards_'+str(rollouts) + 'rollouts.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(y_train, file)

"""
with open('data/x_train_5cards_100rollouts.pkl', 'rb') as file:
    # Call load method to deserialze
    X_train = pickle.load(file)
with open('data/y_train_5cards_100rollouts.pkl', 'rb') as file:
    # Call load method to deserialze
    y_train = pickle.load(file)
"""

epochs = 1000
# Fit the model.
model.fit(X_train, y_train, batch_size=10, epochs=1000, validation_split=0.2)
model.save('models/model'+str(num_cards)+'_'+str(num_cards_on_table)+'cards_'+str(rollouts) + 'rollouts_' + str(epochs) + 'epochs')
