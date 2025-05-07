import os
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import random

from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from q_learn_player import QLearningPlayer

# setup config
CONFIG = {
    "initial_stack": 100000,
    "max_round": 10000,
    "small_blind_amount": 10,
    "n_ehs_bins": 5,
    "use_stack_diff": True,
    "seed": 46,
    "k": 0.5,
    "verbose": True,
}
exp_dir = "output/" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(exp_dir, exist_ok=True)
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

# run game
config = setup_config(max_round=CONFIG["max_round"], initial_stack=CONFIG["initial_stack"], small_blind_amount=CONFIG["small_blind_amount"])
my_player = QLearningPlayer(n_ehs_bins=CONFIG["n_ehs_bins"], is_training=True, use_stack_diff=CONFIG["use_stack_diff"], k=CONFIG["k"])
config.register_player(name="random_player", algorithm=RandomPlayer())
config.register_player(name="q_learn_player", algorithm=my_player)
game_result = start_poker(config, verbose=CONFIG["verbose"])

# save q function and n function
np.save(os.path.join(exp_dir, "q_values.npy"), my_player.Q)
np.save(os.path.join(exp_dir, "n_values.npy"), my_player.N)

# save results 
with open(os.path.join(exp_dir, "results.json"), "w") as f:
    json.dump(my_player.round_results, f, indent=4)

# plot stack and mean reward over time
stack_over_time = [r["stack"] for r in my_player.round_results]
reward_over_time = [r["reward"] for r in my_player.round_results]
mean_reward_over_time = [np.mean(reward_over_time[:i]) for i in range(1, len(reward_over_time))]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(stack_over_time, label="stack", color="orange")
ax2.plot(mean_reward_over_time, label="mean reward", color="blue")
if not CONFIG["use_stack_diff"]:
    ax2.set_ylim(0, 1)
ax1.set_ylim(0, CONFIG["initial_stack"] * 2)
# plot y=INITIAL_STACK line
ax1.plot([0, CONFIG["max_round"]], [CONFIG["initial_stack"], CONFIG["initial_stack"]], color="black", linestyle="--")
ax1.set_xlabel("round")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.savefig(os.path.join(exp_dir, "results.png"))

# how to get final 