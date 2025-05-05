from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator, Event
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck, attach_hole_card
import numpy as np
import deuces
from q_learn_player import (
    REAL_ACTIONS, N_ACTIONS, get_obs, get_exploration_value, update, compute_reward, select_action, _get_a_set
)

def _get_uuid(s):
    game_state, events = s
    return game_state['table'].seats.players[game_state['next_player']].uuid

def _get_n_players(s):
    game_state, events = s
    return len(game_state['table'].seats.players)

def _get_hole_cards(game_state, uuid):
    return [str(card) for card in game_state['table'].seats.players[uuid].hole_card]

def _get_valid_actions(s):
    valid_actions = next(e for e in s[1] if "valid_actions" in e)["valid_actions"]
    return valid_actions


def compute_sim_reward(s):
    game_state, events = s
    uuid = _get_uuid(s)
    # get winners from events
    round_finish_event = next(e for e in events if e["type"] == Event.ROUND_FINISH)
    winners = round_finish_event["winners"]
    round_state = round_finish_event["round_state"]
    return compute_reward(winners, round_state, uuid)


def get_obs_sim(s, n_bins, evaluator):
    uuid = _get_uuid(s)
    hole_cards = _get_hole_cards(s[0], uuid)
    round_state = next(e for e in s[1] if "round_state" in e)["round_state"]
    obs = get_obs(hole_cards, round_state, n_bins, evaluator)
    return obs


def is_terminal(s):
    _gs, events = s
    return any(event["type"] == Event.ROUND_FINISH for event in events)


def search(emulator, s, Q, N, n_rollouts: int):
    for _ in range(n_rollouts):
        out_of_tree = [False for _ in range(_get_n_players(s))]
        simulate(emulator, s, Q, N, out_of_tree)


def simulate(emulator, s, Q, N, evaluator, n_bins, out_of_tree):
    if is_terminal(s):
        # get winners from events?
        return compute_sim_reward(s)
    # check if out of tree
    p_i = s[0]['next_player']
    if out_of_tree[p_i]:
        return rollout(emulator, s, Q, N)
    # get information state
    obs = get_obs_sim(s, n_bins, evaluator)
    # sample action
    a_set = _get_a_set(_get_valid_actions(s))
    if Q[obs[0], obs[1], :].sum() == 0:
        a = np.random.choice(a_set)
        out_of_tree[p_i] = True
    else: 
        a = select_action(Q, N, obs, a_set)
    # get next state
    s = emulator.apply_action(s[0], REAL_ACTIONS[a])
    # sample until terminal
    r = simulate(emulator, s, Q, N, evaluator, n_bins, out_of_tree)
    # backpropagate
    update(Q, N, obs[0], obs[1], a, r)
    return r


def rollout(emulator: Emulator, s, Q, N):
    a_set = _get_a_set(_get_valid_actions(s))
    a = np.random.choice(a_set)
    s = emulator.apply_action(s[0], REAL_ACTIONS[a])
    return simulate(emulator, s, Q, N)


class MCTSPlayer(BasePokerPlayer):

    def __init__(self, n_ehs_bins: int, is_training: bool, k: float=0.5, use_stack_diff: bool=True): 
        self.n_ehs_bins = n_ehs_bins
        # Q[opponent_last_action, hand_strength_bin, action_idx]
        # Q[-1, :, :] is reserved for when you go first
        # TODO: add action history
        self.is_training = is_training
        self.Q = np.zeros((N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.N = np.zeros((N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.k = k
        self.history = []
        
        # logging 
        self.round_results = []
    
    def load_Q(self, Q: np.ndarray | None = None, Q_path: str | None = None):
        if Q is not None:
            self.Q = Q
        elif Q_path is not None:
            self.Q = np.load(Q_path)
        else:
            raise ValueError("No Q provided")
    
    def train(self, n):
        # TODO: sample first 
    
    # TODO: action: simulate at state, then pick argmax action
    # TODO: all the other methdos