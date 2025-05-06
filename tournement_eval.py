from pypokerengine.api.game import setup_config, start_poker
import numpy as np
from mcts_player import MCTSPlayer
from tqdm import tqdm
import json
import os

EPOCHS = 10
EXP_DIR = None # TODO: set
EVAL_DIR = os.path.join(EXP_DIR, "eval")

CONFIG = json.load(open("config.json"))


def create_player(epoch: int):
    epoch_dir = os.path.join(EXP_DIR, f"{epoch}")
    player = MCTSPlayer(
        n_ehs_bins=CONFIG["n_ehs_bins"], 
        is_training=True, 
        n_rollouts_train=CONFIG["n_rollouts_train"],
        n_rollouts_eval=CONFIG["n_rollouts_eval"]
    )
    player.set_emulator(
        player_num=2, 
        max_round=CONFIG["max_round"], 
        small_blind_amount=CONFIG["small_blind_amount"], 
        ante_amount=0, 
        blind_structure={}, 
    )
    player.load(epoch_dir)
    return player



def eval_players(epoch_a, epoch_b):
    player_a = create_player(epoch_a)
    player_b = create_player(epoch_b)

    players_info = {
        f"uuid-{epoch_a}": {
            "stack": CONFIG["initial_stack"],
            "name": f"player_{epoch_a}"
        },
        f"uuid-{epoch_b}": {
            "stack": CONFIG["initial_stack"],
            "name": f"player_{epoch_b}"
        }
    }

    config = setup_config(max_round=CONFIG["max_round"], initial_stack=CONFIG["initial_stack"], small_blind_amount=CONFIG["small_blind_amount"])
    config.register_player(name=f"player_{epoch_a}", algorithm=player_a)
    config.register_player(name=f"player_{epoch_b}", algorithm=player_b)

    game_result = start_poker(config, verbose=1)
    p_a_results = player_a.round_results
    p_b_results = player_b.round_results
    player_a.round_results = []
    player_b.round_results = []
    return p_a_results, p_b_results, game_result


match_up_results = {}
for epoch_a in tqdm(range(EPOCHS), desc="Agent Checkpoint"):
    for epoch_b in tqdm(range(EPOCHS), desc="Other Agent Checkpoint"):
        if epoch_a == epoch_b:
            continue
        p_a_results, p_b_results, game_result = eval_players(epoch_a, epoch_b)
        p_a_mean_reward = np.mean([r["reward"] for r in p_a_results])
        p_b_mean_reward = np.mean([r["reward"] for r in p_b_results])
        p_a_stack = p_a_results[-1]["stack"]
        p_b_stack = p_b_results[-1]["stack"]
        # log results
        match_up_results[(epoch_a, epoch_b)] = {
            "rewards": (p_a_mean_reward, p_b_mean_reward),
            "stacks": (p_a_stack, p_b_stack)
        }
        
        # print results
        print(f"Epoch {epoch_a} vs Epoch {epoch_b}")
        print(f"Player A stack: {p_a_stack}")
        print(f"Player B stack: {p_b_stack}")

# save results
with open(os.path.join(EVAL_DIR, "match_up_results.json"), "w") as f:
    json.dump(match_up_results, f)
