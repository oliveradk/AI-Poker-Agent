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
import random

# Default configuration
N_ROLLOUTS = 0
EVAL_DL = 0
STACK_MARGIN = 2000
DEFAULT_ELO = 1200
K_FACTOR = 32  # Elo K-factor (determines how quickly ratings change)
ELO_DB_FILENAME = "elo_ratings.json"
MATCHUP_LOG_FILENAME = "matchup_performances.csv"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate poker agents using Elo rating system")
    parser.add_argument("--exp_dir", default="output/20250508_154014", help="Experiment directory containing agent checkpoints")
    parser.add_argument("--epochs", type=int, help="Number of epochs to evaluate (default: all available)")
    parser.add_argument("--max_rounds", type=int, default=250, help="Maximum number of rounds per match")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--processes", type=int, default=None, help="Number of parallel processes (default: CPU count)")
    parser.add_argument("--window_size", type=int, default=10, help="Window size for matchup generation")
    return parser.parse_args()

def create_empty_elo_db():
    """Create a new empty Elo database"""
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

def generate_window_matchups(window_epochs):
    """Generate all matchups between epochs in the current window"""
    matchups = []
    for i, epoch_a in enumerate(window_epochs):
        for epoch_b in window_epochs[i+1:]:
            matchups.append((epoch_a, epoch_b))
    return matchups

def generate_previous_matchups(current_epoch, previous_epochs, sample_size, elo_db, default_elo=DEFAULT_ELO):
    """Generate matchups between current epoch and the top-performing previous epochs based on Elo ratings"""
    if not previous_epochs:
        return []
    
    # Sort previous epochs by their Elo ratings (highest first)
    sorted_epochs = sorted(previous_epochs, 
                          key=lambda epoch: elo_db.get(str(epoch), default_elo),
                          reverse=True)
    
    # Take the top sample_size epochs
    top_epochs = sorted_epochs[:sample_size]
    
    return [(current_epoch, prev_epoch) for prev_epoch in top_epochs]

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
    available_epochs.sort()  # Ensure epochs are in ascending order
    
    # Determine which epochs to evaluate
    if args.epochs:
        epochs_to_evaluate = [e for e in available_epochs if e < args.epochs]
    else:
        epochs_to_evaluate = available_epochs
    
    # Create empty Elo database and matchup log
    elo_db_path = os.path.join(eval_dir, ELO_DB_FILENAME)
    elo_db = create_empty_elo_db()
    matchup_log_path = os.path.join(eval_dir, MATCHUP_LOG_FILENAME)
    
    # Initialize for RaisedPlayer
    elo_db["raise"] = DEFAULT_ELO
    
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
    
    # Window-based matchup generation and execution
    window_size = args.window_size
    total_windows = (len(epochs_to_evaluate) + window_size - 1) // window_size
    
    # Add RaisedPlayer to each window
    all_results = []
    
    print(f"Processing {total_windows} windows with size {window_size}")
    
    for window_idx in tqdm(range(total_windows), desc="Processing windows"):
        print(f"\nProcessing window {window_idx+1}/{total_windows}")
        
        # Define current window epochs
        start_idx = window_idx * window_size
        end_idx = min((window_idx + 1) * window_size, len(epochs_to_evaluate))
        window_epochs = epochs_to_evaluate[start_idx:end_idx]
        
        
        # Generate matchups within this window
        window_matchups = generate_window_matchups(window_epochs)
        
        # Generate matchups with previous windows
        previous_epochs = epochs_to_evaluate[:start_idx]
        previous_matchups = []
        
        if previous_epochs:  # Only for windows after the first one
            for epoch in window_epochs:
                sample_size = window_size // 2
                epoch_previous_matchups = generate_previous_matchups(epoch, previous_epochs, sample_size, elo_db)
                previous_matchups.extend(epoch_previous_matchups)
        
        # # Add RaisedPlayer matchups for each epoch in the window
        # raised_matchups = [(epoch, "raise_player") for epoch in window_epochs]
        
        # Combine all matchups
        all_matchups = window_matchups + previous_matchups
        
        # Execute matchups in parallel
        window_results = []
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(all_matchups), desc=f"Window {window_idx+1} matchups") as pbar:
                for result in pool.imap_unordered(worker_fn, all_matchups):
                    if result:  # Skip None results
                        window_results.append(result)
                        
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
                        
                        pbar.update(1)
        
        # Save progress after each window
        save_elo_db(elo_db, elo_db_path)
        all_results.extend(window_results)
        
        # Print window summary
        print(f"\nWindow {window_idx+1} complete. Current Elo ratings:")
        window_agents = [str(e) for e in window_epochs] + ['raise']
        for agent in window_agents:
            print(f"Agent {agent}: {elo_db.get(agent, DEFAULT_ELO):.1f}")
    
    # Print final Elo ratings (sorted)
    print("\nFinal Elo Ratings:")
    sorted_ratings = sorted([(agent, rating) for agent, rating in elo_db.items()], 
                           key=lambda x: x[1], reverse=True)
    for agent, rating in sorted_ratings:
        print(f"Agent {agent}: {rating:.1f}")

if __name__ == "__main__":
    main()
