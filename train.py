import os
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random

from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from mcts_player import MCTSPlayer
# setup config
CONFIG = {
    "initial_stack": 10000,
    "max_round": 500,
    "small_blind_amount": 10,
    "n_ehs_bins": 5,
    "use_stack_diff": False,
    "n_rollouts": 50,
    "seed": 46,
    "k": None,
}
exp_dir = "output/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(exp_dir, exist_ok=True)
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

mtcs_player = MCTSPlayer(
    n_ehs_bins=CONFIG["n_ehs_bins"], 
    is_training=True, 
    use_stack_diff=CONFIG["use_stack_diff"], 
    k=CONFIG["k"],
    n_rollouts=CONFIG["n_rollouts"]
)

mtcs_player.set_emulator(
    player_num=2, 
    max_round=CONFIG["max_round"], 
    small_blind_amount=CONFIG["small_blind_amount"], 
    ante_amount=0, 
    blind_structure={}, 
)

players_info = {
    "uuid-1": {
        "stack": CONFIG["initial_stack"],
        "name": "player_1"
    },
    "uuid-2": {
        "stack": CONFIG["initial_stack"],
        "name": "player_2"
    }
}

mtcs_player.train(n_games=1, players_info=players_info, save_dir=exp_dir)
print("done training")