import matplotlib.pyplot as plt
import json
import os
import pandas as pd
elo_dir = "elo_plots"
k = 0.5
out_dir = "output/unity_out/final_run/20250508_170920/eval"
# k = 1.0 
# out_dir = "output/unity_out/final_run/20250508_184122/eval"
with open(f"{out_dir}/elo_ratings.json", "r") as f:
    elo_ratings = json.load(f)

with open(f"{out_dir}/matchup_performances.csv", "r") as f:
    matchup_performance = pd.read_csv(f)

# compute net winnings against raise
raise_matchups = matchup_performance[matchup_performance["agent_b"] == "raise"]
raise_matchups["net_winnings"] = raise_matchups["a_stack"] - raise_matchups["b_stack"]
mean_winnings = raise_matchups.groupby("agent_a")["net_winnings"].mean().to_dict()


# get elos
epochs = sorted(set(elo_ratings.keys()) - {"raise"})
elos = [elo_ratings[str(epoch)] for epoch in epochs]
games = [i * 100 for i in range(len(epochs))]
raise_elo = elo_ratings["raise"]

# plot elos
plt.figure(figsize=(6, 4.5))
plt.plot(games, elos)
plt.plot(games, [raise_elo] * len(games), "--", label="raise") 
# add label on raise line with the same color as the line
plt.text(games[-1], raise_elo, "raise", ha="right", va="bottom", color=plt.gca().lines[-1].get_color())
plt.xlabel("Simulated Games")
plt.ylabel("Elo")
plt.title("Self-Play Performance Over Training")
plt.savefig(f"{out_dir}/elo_plot_clean.png")
plt.savefig(f"elo_plots/elo_plot_k{k}.svg")
plt.show()

# plot mean winnings against raise
plt.figure(figsize=(6, 4.5))
plt.plot(mean_winnings.keys(), mean_winnings.values())
plt.xlabel("Simulated Games")
plt.ylabel("Mean Winnings vs Raise")
plt.title("Mean Winnings Over Training")
plt.savefig(f"{out_dir}/mean_winnings_plot_clean.png")
plt.savefig(f"elo_plots/mean_winnings_plot_k{k}.svg")
plt.show()
