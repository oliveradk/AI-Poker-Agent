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
    "n_rollouts_train": 10, 
    "n_rollouts_eval": 1,
    "n_games_per_epoch": 2, 
    "n_epochs": 2,
    "n_eval_rounds": 10,
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
    n_rollouts_train=CONFIG["n_rollouts_train"],
    n_rollouts_eval=CONFIG["n_rollouts_eval"]
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


def eval_random_player(player: MCTSPlayer, n_eval_rounds: int):
    config = setup_config(max_round=n_eval_rounds, initial_stack=CONFIG["initial_stack"], small_blind_amount=CONFIG["small_blind_amount"])
    config.register_player(name="random_player", algorithm=RandomPlayer())
    config.register_player(name="my_player", algorithm=player)
    game_result = start_poker(config, verbose=1)
    round_results = player.round_results
    player.round_results = []
    return round_results


# train eval loop
epochs = CONFIG["n_epochs"]
games_per_epoch = CONFIG["n_games_per_epoch"]

mean_eval_reward = []
for epoch in range(epochs):
    mtcs_player.train(n_games=games_per_epoch, players_info=players_info, save_dir=exp_dir)
    print(f"done training epoch {epoch}")

    # evaluate against random player 
    round_results = eval_random_player(mtcs_player, n_eval_rounds=CONFIG["n_eval_rounds"])
    mean_eval_reward.append(np.mean([r["reward"] for r in round_results]))
    with open(os.path.join(exp_dir, f"round_results_{epoch}.json"), "w") as f:
        json.dump(round_results, f)


import matplotlib.pyplot as plt

plt.plot(mean_eval_reward, label="mean eval reward")
plt.legend()
plt.savefig(os.path.join(exp_dir, "mean_eval_reward.png"))

print("done training")