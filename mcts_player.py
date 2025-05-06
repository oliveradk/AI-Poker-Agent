from pypokerengine.players import BasePokerPlayer 
from pypokerengine.api.emulator import Emulator, Event
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck, attach_hole_card
import numpy as np
from randomplayer import RandomPlayer
import os

from q_learn_player import (
    REAL_ACTIONS, N_ACTIONS, get_obs, get_explore_weight, update, compute_reward, select_action, get_a_set
)

def _get_uuid(s):
    game_state, events = s
    return game_state['table'].seats.players[game_state['next_player']].uuid

def _get_n_players(s):
    game_state, events = s
    return len(game_state['table'].seats.players)

def _get_hole_cards(s):
    gs, e = s
    return [str(card) for card in gs['table'].seats.players[gs['next_player']].hole_card]

def _get_valid_actions(s):
    valid_actions = next(e for e in s[1] if "valid_actions" in e)["valid_actions"]
    return valid_actions


def compute_sim_reward(s, use_stack_diff: bool):
    game_state, events = s
    uuid = _get_uuid(s)
    # get winners from events
    round_finish_event = next(e for e in events if e["type"] == Event.ROUND_FINISH)
    winners = round_finish_event["winners"]
    round_state = round_finish_event["round_state"]
    return compute_reward(winners, round_state, uuid, use_stack_diff)


def get_obs_sim(s, n_bins):
    hole_cards = _get_hole_cards(s)
    round_state = next(e for e in s[1] if "round_state" in e)["round_state"]
    obs = get_obs(hole_cards, round_state, n_bins)
    return obs

def get_sim_explore_weight(s, use_stack_diff: bool):
    uuid = _get_uuid(s)
    round_state = next(e for e in s[1] if "round_state" in e)["round_state"]
    return get_explore_weight(round_state, uuid, use_stack_diff)

def get_player(s):
    return s[0]['next_player']


def is_terminal(s):
    _gs, events = s
    return any(event["type"] == Event.ROUND_FINISH for event in events)


def search(emulator, s, Q, N, n_rollouts: int, n_bins, use_stack_diff: bool):
    for _ in range(n_rollouts):
        out_of_tree = [False for _ in range(_get_n_players(s))]
        simulate(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff)


def simulate(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff: bool):
    if is_terminal(s):
        # get winners from events?
        return compute_sim_reward(s, use_stack_diff)
    # check if out of tree
    p_i = get_player(s)
    if out_of_tree[p_i]:
        return rollout(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff)
    # get information state
    obs = get_obs_sim(s, n_bins)
    # sample action
    a_set = get_a_set(_get_valid_actions(s))
    if Q[obs[0], obs[1], :].sum() == 0:
        a = np.random.choice(a_set)
        out_of_tree[p_i] = True
    else: 
        c = get_sim_explore_weight(s, use_stack_diff)
        a = select_action(Q, N, obs, a_set, c)
    # get next state
    s = emulator.apply_action(s[0], REAL_ACTIONS[a])
    # sample until terminal
    r = simulate(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff)
    # backpropagate
    update(Q, N, obs[0], obs[1], a, r)
    return r


def rollout(emulator: Emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff: bool):
    a_set = get_a_set(_get_valid_actions(s))
    a = np.random.choice(a_set)
    s = emulator.apply_action(s[0], REAL_ACTIONS[a])
    return simulate(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff)


class MCTSPlayer(BasePokerPlayer):

    def __init__(self, n_ehs_bins: int, is_training: bool, k: float=0.5, use_stack_diff: bool=True, n_rollouts: int=100): 
        self.n_ehs_bins = n_ehs_bins
        # Q[opponent_last_action, hand_strength_bin, action_idx]
        # Q[-1, :, :] is reserved for when you go first
        # TODO: add action history
        self.is_training = is_training
        self.Q = np.zeros((N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.N = np.zeros((N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.k = k
        self.n_rollouts = n_rollouts
        self.use_stack_diff = use_stack_diff
        self.history = []
        self.emulator = None
        # logging 
        self.round_results = []
    
    def load(self, dir: str):
        self.Q = np.load(os.path.join(dir, "Q.npy"))
        self.N = np.load(os.path.join(dir, "N.npy"))
    
    def save(self, dir: str):
        np.save(os.path.join(dir, "Q.npy"), self.Q)
        np.save(os.path.join(dir, "N.npy"), self.N)

    
    def set_emulator(self, player_num, max_round, small_blind_amount, ante_amount, blind_structure):
        self.emulator = Emulator()
        self.emulator.set_game_rule(player_num, max_round, small_blind_amount, ante_amount)
        self.emulator.set_blind_structure(blind_structure)
        # for info in players_info: # don't think i actually need this, since I never call run until finish
        #     self.emulator.register_player(info["uuid"], RandomPlayer())

    # Setup Emulator object by registering game information
    def receive_game_start_message(self, game_info):
        player_num = game_info["player_num"]
        max_round = game_info["rule"]["max_round"]
        small_blind_amount = game_info["rule"]["small_blind_amount"]
        ante_amount = game_info["rule"]["ante"]
        blind_structure = game_info["rule"]["blind_structure"]
        players_info = game_info["seats"]["players"]
        self.players_info = players_info
        self.set_emulator(player_num, max_round, small_blind_amount, ante_amount, blind_structure, players_info)
    
    # TODO: what are we doing
    def train(self, n_games, players_info, save_dir: str):
        if self.emulator is None:
            raise ValueError("Emulator not set")

        initial_state = self.emulator.generate_initial_game_state(players_info)
        for i in range(n_games):
            s = self.emulator.start_new_round(initial_state)
            search(self.emulator, s, self.Q, self.N, self.n_rollouts, self.n_ehs_bins, self.use_stack_diff)
        
        self.save(save_dir)