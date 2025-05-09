from pypokerengine.api.game import setup_config, start_poker
from mcts_player import MCTSPlayer
from raise_player import RaisedPlayer
from tqdm import tqdm
import json
import os
import argparse
import math
import matplotlib.pyplot as plt
import csv
import multiprocessing
from functools import partial
from multiprocessing import Manager, Lock

# TODO: continually test againt player queue and raise player
# Default configuration
N_ROLLOUTS = 0
EVAL_DL = 0
STACK_MARGIN = 2000
DEFAULT_ELO = 1200
K_FACTOR = 32  # Elo K-factor (determines how quickly ratings change)
MIN_MATCHUPS = 5  # Minimum number of matchups for new agents
ELO_DB_FILENAME = "elo_ratings.json"
MATCHUP_LOG_FILENAME = "matchup_performances.csv"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate poker agents using Elo rating system")
    parser.add_argument("--exp_dir", default="output/20250508_154014", help="Experiment directory containing agent checkpoints")
    parser.add_argument("--epochs", type=int, help="Number of epochs to evaluate (default: all available)")
    parser.add_argument("--max_rounds", type=int, default=250, help="Maximum number of rounds per match")
    parser.add_argument("--verbose", type=int, default=True, help="Verbosity level")
    parser.add_argument("--processes", type=int, default=None, help="Number of parallel processes (default: CPU count)")
    return parser.parse_args()

def load_or_create_elo_db(elo_db_path):
    if os.path.exists(elo_db_path):
        with open(elo_db_path, "r") as f:
            return json.load(f)
    return {}

def save_elo_db(elo_db, elo_db_path):
    with open(elo_db_path, "w") as f:
        json.dump(elo_db, f, indent=2)

def expected_score(rating_a, rating_b):
    """Calculate expected score (probability of winning) for player A against player B"""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

def update_elo(rating_a, rating_b, score_a, k_factor=K_FACTOR):
    """Update Elo ratings based on match outcome
    score_a should be 1 for A win, 0 for B win, 0.5 for draw
    """
    expected_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - score_a) - (1 - expected_a))
    return new_rating_a, new_rating_b

def determine_outcome(p_a_stack, p_b_stack):
    """Determine match outcome based on final stacks"""
    if p_a_stack > p_b_stack + STACK_MARGIN:
        return 1.0  # Player A wins
    elif p_b_stack > p_a_stack + STACK_MARGIN:
        return 0.0  # Player B wins
    else:
        return 0.5  # Draw

def select_matchups(epochs, elo_db, new_epochs=None, n_matchups=None):
    """Select matchups intelligently for accurate Elo evaluation
    
    For new agents: match against agents with well-established ratings
    For all agents: prioritize matchups with agents of similar strength
    """
    all_agents = [f"{e}" for e in epochs]
    matchups = []
    
    # If new_epochs is provided, focus on evaluating those
    if new_epochs:
        established_agents = [a for a in all_agents if a in elo_db and a not in [f"{e}" for e in new_epochs]]
        
        for new_epoch in new_epochs:
            new_agent = f"{new_epoch}"
            
            # Ensure new agent plays against a diverse set of opponents
            if established_agents:
                # Sort established agents by Elo
                sorted_established = sorted(established_agents, key=lambda a: elo_db.get(a, DEFAULT_ELO))
                
                # Select opponents across the Elo spectrum for calibration
                num_opponents = min(MIN_MATCHUPS, len(established_agents))
                step = max(1, len(established_agents) // num_opponents)
                selected_opponents = [sorted_established[i] for i in range(0, len(sorted_established), step)][:num_opponents]
                
                for opponent in selected_opponents:
                    matchups.append((int(new_agent), int(opponent)))
            
            # Also match against other new agents
            for other_new in [f"{e}" for e in new_epochs]:
                if other_new != new_agent:
                    matchups.append((int(new_agent), int(other_new)))
            
            # Add RaisedPlayer matchup for every agent
            matchups.append((int(new_agent), "raise_player"))
    return matchups

def create_player(epoch, exp_dir, config):
    epoch_dir = os.path.join(exp_dir, f"{epoch}")
    player = MCTSPlayer(
        n_ehs_bins=config["n_ehs_bins"], 
        n_rollouts_train=config["n_rollouts_train"],
        n_rollouts_eval=N_ROLLOUTS, 
        eval_dl=EVAL_DL, 
    )
    player.set_emulator(
        player_num=2, 
        max_round=float("inf"),
        small_blind_amount=config["small_blind_amount"], 
        ante_amount=0, 
        blind_structure={}, 
    )
    player.load(epoch_dir)
    return player

def eval_players(epoch_a, epoch_b, rounds, exp_dir, config, verbose, player_b_type="custom"):
    player_a = create_player(epoch_a, exp_dir, config)
    
    if player_b_type == "custom":
        player_b = create_player(epoch_b, exp_dir, config)
        player_b_name = f"{epoch_b}"
    elif player_b_type == "raise":
        player_b = RaisedPlayer()
        player_b_name = "raise"
    
    game_config = setup_config(max_round=rounds, initial_stack=config["initial_stack"], small_blind_amount=config["small_blind_amount"])
    game_config.register_player(name=f"player_{epoch_a}", algorithm=player_a)
    game_config.register_player(name=f"player_{player_b_name}", algorithm=player_b)

    game_result = start_poker(game_config, verbose=verbose)
    p_a_stack = game_result["players"][0]["stack"]
    p_b_stack = game_result["players"][1]["stack"]

    return p_a_stack, p_b_stack

def log_matchup_performance(log_file, agent_a, agent_b, p_a_stack, p_b_stack, outcome):
    """Log matchup performance to CSV file"""
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as csvfile:
        fieldnames = ['agent_a', 'agent_b', 'a_stack', 'b_stack', 'outcome']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'agent_a': agent_a,
            'agent_b': agent_b,
            'a_stack': p_a_stack,
            'b_stack': p_b_stack,
            'outcome': outcome
        })

def evaluate_single_matchup(matchup, max_rounds, exp_dir, config, verbose):
    """Worker function to evaluate a single matchup in a separate process"""
    epoch_a, epoch_b = matchup
    
    # Skip self-play
    if epoch_a == epoch_b:
        return None
    
    # Determine player type
    player_b_type = "raise" if epoch_b == "raise_player" else "custom"
    
    # Evaluate the matchup
    p_a_stack, p_b_stack = eval_players(
        epoch_a, epoch_b, max_rounds, exp_dir, config, verbose, 
        player_b_type=player_b_type
    )
    
    agent_a = f"{epoch_a}"
    agent_b = f"{epoch_b}" if player_b_type == "custom" else "raise"
    
    # Determine outcome for Elo calculation
    outcome = determine_outcome(p_a_stack, p_b_stack)
    
    return {
        'matchup': matchup,
        'agent_a': agent_a,
        'agent_b': agent_b,
        'p_a_stack': p_a_stack,
        'p_b_stack': p_b_stack,
        'outcome': outcome
    }

def main():
    args = parse_arguments()
    
    # Setup directories
    exp_dir = args.exp_dir
    eval_dir = os.path.join(exp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load configuration
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    # Determine available epochs
    available_epochs = [int(d) for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d)) and d.isdigit()]
    # filter out epochs that don't have a model checkpoint
    available_epochs = [e for e in available_epochs if os.path.exists(os.path.join(exp_dir, f"{e}", "Q.npy"))]
    
    # Determine which epochs to evaluate
    if args.epochs:
        max_epoch = args.epochs
        epochs_to_evaluate = [e for e in available_epochs if e < max_epoch]
    else:
        max_epoch = max(available_epochs) + 1
        epochs_to_evaluate = available_epochs
    
    # Load or create Elo database
    elo_db_path = os.path.join(eval_dir, ELO_DB_FILENAME)
    elo_db = load_or_create_elo_db(elo_db_path)
    
    # Set up matchup log file
    matchup_log_path = os.path.join(eval_dir, MATCHUP_LOG_FILENAME)
    
    # Identify new epochs (not in matchup_performances.csv)
    existing_epochs = []
    if os.path.exists(matchup_log_path):
        with open(matchup_log_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            # Extract unique agent_a values that are digits and convert to int
            existing_epochs = set(int(row['agent_a']) for row in reader 
                                if row['agent_a'].isdigit())
    
    new_epochs = [epoch for epoch in epochs_to_evaluate if epoch not in existing_epochs]
    
    print(f"Evaluating epochs: {epochs_to_evaluate}")
    print(f"New epochs to evaluate: {new_epochs}")
    
    # Select matchups for evaluation
    if new_epochs:
        # More matchups for new agents to establish accurate ratings
        n_matchups = len(new_epochs) * MIN_MATCHUPS
    else:
        # Fewer matchups if just updating existing ratings
        n_matchups = len(epochs_to_evaluate) // 2
    
    matchups = select_matchups(epochs_to_evaluate, elo_db, new_epochs, n_matchups)
    print(f"Selected {len(matchups)} matchups for evaluation")
    
    # Set up multiprocessing
    num_processes = args.processes if args.processes else multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for parallel evaluation")
    
    # Create a partial function with all the fixed arguments
    worker_fn = partial(
        evaluate_single_matchup, 
        max_rounds=args.max_rounds, 
        exp_dir=exp_dir, 
        config=config, 
        verbose=0  # Set to 0 to prevent output clutter in parallel processes
    )
    
    # Set up multiprocessing with shared resources
    manager = Manager()
    elo_lock = Lock()  # Lock for synchronizing Elo database updates
    save_counter = manager.Value('i', 0)  # Shared counter for tracking when to save
    save_frequency = max(1, len(matchups) // 10)  # Save after every ~10% of matchups
    
    # Run matchups in parallel
    print("\nEvaluating all matchups in parallel and updating Elo as results arrive:")
    all_results = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap_unordered to get results as they complete
        with tqdm(total=len(matchups), desc="Evaluating matchups") as pbar:
            for result in pool.imap_unordered(worker_fn, matchups):
                if result:  # Skip None results (self-play matchups)
                    all_results.append(result)
                    
                    # Update Elo ratings as results come in
                    agent_a = result['agent_a']
                    agent_b = result['agent_b']
                    p_a_stack = result['p_a_stack']
                    p_b_stack = result['p_b_stack']
                    outcome = result['outcome']
                    
                    # Log matchup performance
                    log_matchup_performance(
                        matchup_log_path, agent_a, agent_b, p_a_stack, p_b_stack, outcome
                    )
                    
                    # Use lock to safely update the Elo database
                    with elo_lock:
                        # Get current Elo ratings
                        rating_a = elo_db.get(agent_a, DEFAULT_ELO)
                        rating_b = elo_db.get(agent_b, DEFAULT_ELO)
                        
                        # Update Elo ratings
                        new_rating_a, new_rating_b = update_elo(rating_a, rating_b, outcome)
                        elo_db[agent_a] = new_rating_a
                        elo_db[agent_b] = new_rating_b
                        
                        # Increment save counter and save periodically
                        save_counter.value += 1
                        if save_counter.value % save_frequency == 0:
                            save_elo_db(elo_db, elo_db_path)
                    
                    # Print results (outside the lock to not block other processes)
                    print(f"\nMatch: {agent_a} vs {agent_b}")
                    print(f"Final stacks: {agent_a}={p_a_stack}, {agent_b}={p_b_stack}")
                    print(f"Elo before: {agent_a}={rating_a:.1f}, {agent_b}={rating_b:.1f}")
                    print(f"Elo after: {agent_a}={new_rating_a:.1f}, {agent_b}={new_rating_b:.1f}")
                    print(f"Completed {save_counter.value}/{len(matchups)} matchups")
                    
                    pbar.update(1)
    
    # Make sure the final Elo database is saved
    save_elo_db(elo_db, elo_db_path)
    
    # Print final Elo ratings (sorted)
    print("\nFinal Elo Ratings:")
    sorted_ratings = sorted([(agent, rating) for agent, rating in elo_db.items()], 
                           key=lambda x: x[1], reverse=True)
    for agent, rating in sorted_ratings:
        print(f"Agent {agent}: {rating:.1f}")
    
    # Plot Elo ratings
    agents = [a for a, _ in sorted_ratings]
    ratings = [r for _, r in sorted_ratings]
    
    plt.figure(figsize=(10, 6))
    plt.bar(agents, ratings)
    plt.axhline(y=DEFAULT_ELO, color='r', linestyle='--', alpha=0.3, label=f'Default ({DEFAULT_ELO})')
    plt.xlabel("Agent")
    plt.ylabel("Elo Rating")
    plt.title("Agent Elo Ratings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "elo_ratings.png"))
    
    # Also plot Elo progression over epochs
    epoch_ratings = [(int(agent), rating) for agent, rating in elo_db.items() if agent.isdigit()]
    epoch_ratings.sort(key=lambda x: x[0])  # Sort by epoch
    
    if epoch_ratings:
        epochs = [e for e, _ in epoch_ratings]
        ratings = [r for _, r in epoch_ratings]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, ratings, marker='o')
        plt.axhline(y=DEFAULT_ELO, color='r', linestyle='--', alpha=0.3, label=f'Default ({DEFAULT_ELO})')
        plt.xlabel("Epoch")
        plt.ylabel("Elo Rating")
        plt.title("Elo Rating Progression Over Epochs")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "elo_progression.png"))
        
        # Also plot performance against RaisedPlayer
        if os.path.exists(matchup_log_path):
            with open(matchup_log_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                raised_matchups = [row for row in reader if row['agent_b'] == 'raise']
                
                if raised_matchups:
                    # Group by epoch and calculate average stack difference
                    raise_results = {}
                    for row in raised_matchups:
                        epoch = int(row['agent_a'])
                        stack_diff = int(row['a_stack']) - int(row['b_stack'])
                        if epoch in raise_results:
                            raise_results[epoch].append(stack_diff)
                        else:
                            raise_results[epoch] = [stack_diff]
                    
                    # Calculate average stack difference per epoch
                    epochs = sorted(raise_results.keys())
                    avg_stack_diffs = [sum(raise_results[e])/len(raise_results[e]) for e in epochs]
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(epochs, avg_stack_diffs, marker='o')
                    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                    plt.xlabel("Epoch")
                    plt.ylabel("Avg Stack Difference vs RaisedPlayer")
                    plt.title("Performance Against RaisedPlayer Over Epochs")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(eval_dir, "raised_player_performance.png"))

if __name__ == "__main__":
    main()
