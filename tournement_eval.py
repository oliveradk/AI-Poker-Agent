from pypokerengine.api.game import setup_config, start_poker
import numpy as np
from mcts_player import MCTSPlayer
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from tqdm import tqdm
import json
import os
import sys
import itertools

MAX_ROUNDS = 500
N_ROLLOUTS = 0
VERBOSE = 1

# output/20250506_185704 - severe cycling
EXP_DIR = sys.argv[1]
EPOCHS = 5 #max([int(d) for d in os.listdir(EXP_DIR) if os.path.isdir(os.path.join(EXP_DIR, d)) and d.isdigit()]) + 1
EVAL_DIR = os.path.join(EXP_DIR, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)
CONFIG = json.load(open(os.path.join(EXP_DIR, "config.json")))


def create_player(epoch: int):
    epoch_dir = os.path.join(EXP_DIR, f"{epoch}")
    player = MCTSPlayer(
        n_ehs_bins=CONFIG["n_ehs_bins"], 
        n_rollouts_train=CONFIG["n_rollouts_train"],
        n_rollouts_eval=N_ROLLOUTS, 
        eval_dl=CONFIG["eval_dl"]
    )
    player.set_emulator(
        player_num=2, 
        max_round=float("inf"), #TODO: should change 
        small_blind_amount=CONFIG["small_blind_amount"], 
        ante_amount=0, 
        blind_structure={}, 
    )
    player.load(epoch_dir)
    return player



def eval_players(epoch_a, epoch_b, rounds):
    player_a = create_player(epoch_a)
    player_b = create_player(epoch_b)

    config = setup_config(max_round=rounds, initial_stack=CONFIG["initial_stack"], small_blind_amount=CONFIG["small_blind_amount"])
    config.register_player(name=f"player_{epoch_a}", algorithm=player_a)
    config.register_player(name=f"player_{epoch_b}", algorithm=player_b)

    game_result = start_poker(config, verbose=VERBOSE)
    p_a_results = player_a.round_results
    p_b_results = player_b.round_results
    player_a.round_results = []
    player_b.round_results = []
    return p_a_results, p_b_results, game_result


# TODO: why is this glitching and infinitely raising?
match_up_results = []
matchups = list(itertools.combinations(range(EPOCHS), 2))
for matchups in tqdm(matchups, desc="Agent Checkpoint"):
    epoch_a, epoch_b = matchups
    if epoch_a == epoch_b:
        continue
    p_a_results, p_b_results, game_result = eval_players(epoch_a, epoch_b, MAX_ROUNDS)
    p_a_mean_reward = np.mean([r["reward"] for r in p_a_results])
    p_b_mean_reward = np.mean([r["reward"] for r in p_b_results])
    p_a_stack = p_a_results[-1]["stack"]
    p_b_stack = p_b_results[-1]["stack"]
    rounds = len(p_a_results)
    # log results
    match_up_results.append({
        "p1": epoch_a,  
        "p2": epoch_b,
        "rewards": [p_a_mean_reward, p_b_mean_reward],
        "stacks": [p_a_stack, p_b_stack],
        "rounds": rounds
    })
    
    # print results
    print(f"{epoch_a} vs {epoch_b}")
    print(f"{epoch_a} stack: {p_a_stack}")
    print(f"{epoch_b} stack: {p_b_stack}")
    print(f"{epoch_a} rounds: {rounds}")

# save results
with open(os.path.join(EVAL_DIR, "match_up_results.json"), "w") as f:
    json.dump(match_up_results, f)

# plot epoch and average reward 
avg_match_up_rewards = []
for epoch in range(EPOCHS):
    epoch_match_results = [result for result in match_up_results if result["p1"] == epoch or result["p2"] == epoch]
    idxs = [result["p2"] == epoch for result in epoch_match_results]
    avg_match_up_rewards.append(np.mean([result["rewards"][i] for i, result in zip(idxs, epoch_match_results)]))

# plot epoch and average reward 
import matplotlib.pyplot as plt
plt.plot(avg_match_up_rewards)
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.savefig(os.path.join(EVAL_DIR, "epoch_vs_avg_reward.png"))
plt.show()
