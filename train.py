import os
import json
import numpy as np
import datetime as dt
import random

from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from mcts_player import MCTSPlayer
# setup config
CONFIG = {
    "initial_stack": 10000,
    "small_blind_amount": 10,
    "n_ehs_bins": 5,
    "n_rollouts_train": 100, # TODO: 100
    "n_rollouts_eval": 5,
    "eval_dl": 2,
    "n_games_per_epoch": 10, 
    "n_epochs": 1000,
    "n_eval_rounds": 0,
    "seed": 42,
    "load_dir": None, 
    "eval_oppo": "random_player",
    "verbose": True
}
exp_dir = "output/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(exp_dir, exist_ok=True)
with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(CONFIG, f)
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

mtcs_player = MCTSPlayer(
    n_ehs_bins=CONFIG["n_ehs_bins"], 
    n_rollouts_train=CONFIG["n_rollouts_train"],
    n_rollouts_eval=CONFIG["n_rollouts_eval"], 
    eval_dl=CONFIG["eval_dl"]
)

mtcs_player.set_emulator(
    player_num=2, 
    max_round=float("inf"),
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

def eval_against_player(player: MCTSPlayer, n_eval_rounds: int, oppo, oppo_name, verbose: int=0,):
    config = setup_config(max_round=n_eval_rounds, initial_stack=CONFIG["initial_stack"], small_blind_amount=CONFIG["small_blind_amount"])
    config.register_player(name=oppo_name, algorithm=oppo)
    config.register_player(name="my_player", algorithm=player)
    game_result = start_poker(config, verbose=verbose)
    round_results = player.round_results
    player.round_results = []
    return round_results, game_result

def eval_against_random_player(player: MCTSPlayer, n_eval_rounds: int, verbose: int=0):
    return eval_against_player(player, n_eval_rounds, RandomPlayer(), "random_player", verbose)

def eval_against_raise_player(player: MCTSPlayer, n_eval_rounds: int, verbose: int=0):
    return eval_against_player(player, n_eval_rounds, RaisedPlayer(), "raise_player", verbose)

# train eval loop
epochs = CONFIG["n_epochs"]
games_per_epoch = CONFIG["n_games_per_epoch"]

oppo = RaisedPlayer() if CONFIG["eval_oppo"] == "raise_player" else RandomPlayer()
if CONFIG["load_dir"] is not None:
    mtcs_player.load(CONFIG["load_dir"])

    round_results, game_result = eval_against_player(
        mtcs_player, n_eval_rounds=CONFIG["n_eval_rounds"], oppo=oppo, oppo_name=CONFIG["eval_oppo"], verbose=1
    )

for epoch in range(epochs):
    # create epoch dir
    epoch_dir = os.path.join(exp_dir, f"{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # train
    mtcs_player.train(n_games=games_per_epoch, players_info=players_info, save_dir=epoch_dir)
    print(f"Epoch {epoch} done")

    # eval against random player 
    round_results, game_result = eval_against_player(
        mtcs_player, n_eval_rounds=CONFIG["n_eval_rounds"], oppo=oppo, oppo_name=CONFIG["eval_oppo"], verbose=CONFIG["verbose"]
    )

    # reload weights (to undo any changes from eval)
    mtcs_player.load(epoch_dir)

    
    # save results
    with open(os.path.join(epoch_dir, f"round_results.json"), "w") as f:
        json.dump(round_results, f)