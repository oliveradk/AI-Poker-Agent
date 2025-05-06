from pypokerengine.players import BasePokerPlayer 
from pypokerengine.api.emulator import Emulator, Event
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.action_checker import ActionChecker
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck, attach_hole_card
import numpy as np
import os
from tqdm import tqdm

from q_learn_player import (
    REAL_ACTIONS,
    N_ACTIONS,
    N_STREETS,
    get_obs,
    get_explore_weight,
    update,
    compute_reward,
    select_action,
    get_a_set, 
    get_round_log
)

def _get_uuid(s):
    state, events = s
    return state['table'].seats.players[state['next_player']].uuid

def _get_n_players(s):
    state, events = s
    return len(state['table'].seats.players)

def _get_hole_cards(s):
    state, events = s
    return [str(card) for card in state['table'].seats.players[state['next_player']].hole_card]

def _get_valid_actions(s): #TODO:
    state, events = s
    if len(events) > 0:
        return next(e for e in events if "valid_actions" in e)["valid_actions"]
    # encode using game state
    players = state["table"].seats.players
    player_pos = state["next_player"]
    valid_actions = ActionChecker.legal_actions(players, player_pos, state["small_blind_amount"],state["street"])
    return valid_actions


def compute_sim_rewards(s, use_stack_diff: bool):
    state, events = s
    uuids = [p.uuid for p in state["table"].seats.players]
    # get winners from events
    round_finish_event = next(e for e in events if e["type"] == Event.ROUND_FINISH)
    winners = round_finish_event["winners"]
    round_state = round_finish_event["round_state"]
    rewards = [compute_reward(winners, round_state, uuid, use_stack_diff) for uuid in uuids] # TODO: make more efficient
    return rewards


def get_obs_sim(s, n_bins):
    hole_cards = _get_hole_cards(s)
    round_state = DataEncoder.encode_round_state(s[0])
    uuid = _get_uuid(s)
    obs = get_obs(hole_cards, round_state, uuid, n_bins)
    return obs

def get_sim_explore_weight(s, use_stack_diff: bool):
    uuid = _get_uuid(s)
    round_state = DataEncoder.encode_round_state(s[0])
    return get_explore_weight(round_state, uuid, use_stack_diff)

def get_player(s):
    return s[0]['next_player']


def is_terminal(s):
    state, events = s
    return any(event["type"] == Event.ROUND_FINISH for event in events)


def search(emulator, s, Q, N, n_rollouts: int, n_bins, use_stack_diff: bool):
    for i in range(n_rollouts):
        out_of_tree = [False for _ in range(_get_n_players(s))]
        simulate(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff)


def simulate(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff: bool):
    if is_terminal(s):
        # get winners from events?
        return compute_sim_rewards(s, use_stack_diff)
    # check if out of tree
    p_i = get_player(s)
    if out_of_tree[p_i]:
        return rollout(emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff)
    # get information state
    obs = get_obs_sim(s, n_bins)
    # sample action
    a_set = get_a_set(_get_valid_actions(s))
    if Q[obs].sum() == 0:
        a = np.random.choice(a_set)
        out_of_tree[p_i] = True
    else: 
        c = get_sim_explore_weight(s, use_stack_diff)
        a = select_action(Q, N, obs, a_set, c)
    # get next state
    s_next = emulator.apply_action(s[0], REAL_ACTIONS[a])
    # sample until terminal
    rs = simulate(emulator, s_next, Q, N, out_of_tree, n_bins, use_stack_diff)
    # backpropagate
    update(Q, N, obs + (a,), rs[p_i])
    return rs


def rollout(emulator: Emulator, s, Q, N, out_of_tree, n_bins, use_stack_diff: bool):
    a_set = get_a_set(_get_valid_actions(s))
    a = np.random.choice(a_set)
    s_next = emulator.apply_action(s[0], REAL_ACTIONS[a])
    return simulate(emulator, s_next, Q, N, out_of_tree, n_bins, use_stack_diff)


# TODO: add action history 
# TODO: add "small blind based" rewards
# TODO: figure out how to eval against actual opponents
# # nah I'll just evaluate against older epochs, seems reasonable enough


class MCTSPlayer(BasePokerPlayer):
    # based on https://cdn.aaai.org/ocs/ws/ws1227/8811-38072-1-PB.pdf

    def __init__(
            self, 
            n_ehs_bins: int, 
            is_training: bool, 
            k: float=0.5, 
            use_stack_diff: bool=True, 
            n_rollouts_train: int=100,
            n_rollouts_eval: int=100
        ): 
        self.n_ehs_bins = n_ehs_bins
        # Q[position, round, last_action, hand_strength_bin, action_idx]
        # Q[-1, :, :] is reserved for when you go first
        # TODO: add action history
        self.is_training = is_training
        self.Q = np.zeros((2, N_STREETS, N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.N = np.zeros((2, N_STREETS, N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.k = k
        self.n_rollouts_train = n_rollouts_train
        self.n_rollouts_eval = n_rollouts_eval
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
    
    # TODO: what are we doing
    def train(self, n_games, players_info, save_dir: str, log_interval: int=100):
        if self.emulator is None:
            raise ValueError("Emulator not set")

        initial_state = self.emulator.generate_initial_game_state(players_info)
        for i in tqdm(range(n_games), desc="Training games"):
            s = self.emulator.start_new_round(initial_state)
            search(self.emulator, s, self.Q, self.N, self.n_rollouts_train, self.n_ehs_bins, self.use_stack_diff)
            if i % log_interval == 0:
                # print mean q values over first 2 axes (so you get q values for each action)
                mean_q_values = np.mean(self.Q, axis=tuple(range(self.Q.ndim-1)))
                mean_n_values = np.mean(self.N, axis=tuple(range(self.N.ndim-1)))
                assert len(mean_q_values) == len(mean_n_values) == N_ACTIONS
                print(f"Mean Q values: {[f'{REAL_ACTIONS[i]}: {mean_q_values[i]:.2f}' for i in range(N_ACTIONS)]}")
                print(f"Mean N values: {[f'{REAL_ACTIONS[i]}: {mean_n_values[i]:.2f}' for i in range(N_ACTIONS)]}")
        self.save(save_dir)

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
    

    def declare_action(self, valid_actions, hole_card, round_state):
        if len(valid_actions) == 1: # not a real action if no choice
           return valid_actions[0]["action"]
        # get game state from round state 
        game_state = restore_game_state(round_state)
        for player in game_state["table"].seats.players:
           game_state = attach_hole_card_from_deck(game_state, player.uuid)
        # simulate for alloted budget 
        search(self.emulator, (game_state, []), self.Q, self.N, self.n_rollouts_eval, self.n_ehs_bins, self.use_stack_diff)
        # argmax best action
        obs = get_obs(hole_card, round_state, self.uuid, self.n_ehs_bins)
        a = np.argmax(self.Q[obs])
        return REAL_ACTIONS[a]

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        reward = compute_reward(winners, round_state, self.uuid, self.use_stack_diff)
        
        # update Q function, N function
        self._log_result(winners, hand_info, round_state, reward)
    
    def _log_result(self, winners, hand_info, round_state, reward):
        round_log = get_round_log(round_state, reward, self.uuid)
        self.round_results.append(round_log)
    
