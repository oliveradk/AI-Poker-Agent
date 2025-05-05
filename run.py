from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from our_player import MCTSPlayer

config = setup_config(max_round=1, initial_stack=10000, small_blind_amount=10)

config.register_player(name="random_player", algorithm=RandomPlayer())
config.register_player(name="mcts_player", algorithm=MCTSPlayer(n_ehs_bins=10, is_training=True))

game_result = start_poker(config, verbose=1)

print(game_result)

# how to get final 