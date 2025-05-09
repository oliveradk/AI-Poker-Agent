from pypokerengine.players import BasePokerPlayer 
from pypokerengine.api.emulator import Emulator, Event
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.action_checker import ActionChecker
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck, attach_hole_card
import numpy as np
import os
from tqdm import tqdm
import json

from q_learn_player import (
    REAL_ACTIONS,
    N_ACTIONS,
    N_STREETS,
    init_Q_and_N,
    load_Q_and_N,
    save_Q_and_N,
    get_obs,
    get_explore_weight,
    update,
    compute_reward,
    select_action,
    get_a_set, 
    get_round_log,
    _get_stack
)

USE_STACK_DIFF = True

def _get_uuid(s):
    state, events = s
    return state['table'].seats.players[state['next_player']].uuid

def _get_n_players(s):
    state, events = s
    return len(state['table'].seats.players)

def _get_hole_cards(s):
    state, events = s
    return [str(card) for card in state['table'].seats.players[state['next_player']].hole_card]

def _get_valid_actions(s):
    state, events = s
    if len(events) > 0:
        return next(e for e in events if "valid_actions" in e)["valid_actions"]
    # encode using game state
    players = state["table"].seats.players
    player_pos = state["next_player"]
    valid_actions = ActionChecker.legal_actions(players, player_pos, state["small_blind_amount"],state["street"])
    return valid_actions

def _get_init_stacks(s):
    state, events = s
    return [p.stack for p in state["table"].seats.players]


def compute_sim_rewards(s, init_stacks):
    state, events = s
    uuids = [p.uuid for p in state["table"].seats.players]
    # get winners from events
    round_finish_event = next(e for e in events if e["type"] == Event.ROUND_FINISH)
    winners = round_finish_event["winners"]
    round_state = round_finish_event["round_state"]
    rewards = [compute_reward(winners, round_state, uuid, init_stack, USE_STACK_DIFF) 
               for uuid, init_stack in zip(uuids, init_stacks)] # TODO: make more efficient
    return rewards


def get_obs_sim(s, n_bins):
    hole_cards = _get_hole_cards(s)
    round_state = DataEncoder.encode_round_state(s[0])
    uuid = _get_uuid(s)
    obs = get_obs(hole_cards, round_state, uuid, n_bins)
    return obs

def get_sim_explore_weight(s, k):
    uuid = _get_uuid(s)
    round_state = DataEncoder.encode_round_state(s[0])
    return get_explore_weight(round_state, uuid, USE_STACK_DIFF, k=k)

def get_player(s):
    return s[0]['next_player']


def is_terminal(s):
    state, events = s
    return any(event["type"] == Event.ROUND_FINISH for event in events)


def search(emulator, s, Q, N, n_rollouts: int, n_bins, init_stacks, depth_limit, k):
    for i in range(n_rollouts):
        out_of_tree = np.zeros(_get_n_players(s), dtype=bool)
        simulate(emulator, s, Q, N, out_of_tree, n_bins, init_stacks, depth_limit, k)


def simulate(emulator, s, Q, N, out_of_tree, n_bins, init_stacks, depth_limit, k):
    if is_terminal(s):
        # get winners from events?
        return compute_sim_rewards(s, init_stacks)
    if depth_limit == 0:
        out_of_tree[:] = True
    else: 
        depth_limit -= 1
    # check if out of tree
    p_i = get_player(s)
    if out_of_tree[p_i]:
        return rollout(emulator, s, Q, N, out_of_tree, n_bins, init_stacks, depth_limit, k)
    # get information state
    obs = get_obs_sim(s, n_bins)
    # sample action
    a_set = get_a_set(_get_valid_actions(s))
    if Q[obs].sum() == 0:
        a = np.random.choice(a_set)
        out_of_tree[p_i] = True
    else: 
        c = get_sim_explore_weight(s, k)
        a = select_action(Q, N, obs, a_set, c)
    # get next state
    s_next = emulator.apply_action(s[0], REAL_ACTIONS[a])
    # sample until terminal
    rs = simulate(emulator, s_next, Q, N, out_of_tree, n_bins, init_stacks, depth_limit, k)
    # backpropagate
    update(Q, N, obs + (a,), rs[p_i])
    return rs


def rollout(emulator: Emulator, s, Q, N, out_of_tree, n_bins, init_stacks, depth_limit, k):
    a_set = get_a_set(_get_valid_actions(s))
    a = np.random.choice(a_set)
    s_next = emulator.apply_action(s[0], REAL_ACTIONS[a])
    return simulate(emulator, s_next, Q, N, out_of_tree, n_bins, init_stacks, depth_limit, k)



class MCTSPlayer(BasePokerPlayer):
    # based on https://cdn.aaai.org/ocs/ws/ws1227/8811-38072-1-PB.pdf

    def __init__(
        self, 
        n_ehs_bins: int | None = None,
        n_rollouts_train: int | None = None,
        n_rollouts_eval: int | None = None,
        eval_dl: int | None = None,
        k: float | None = None
    ): 
        from_config = n_ehs_bins is None
        if from_config:
            # load from config
            with open("config.json") as f:
                config = json.load(f)
            n_ehs_bins = config["n_ehs_bins"]
            n_rollouts_train = config["n_rollouts_train"]
            n_rollouts_eval = config["n_rollouts_eval"]
            eval_dl = config["eval_dl"]
            k = config["k"]
        self.n_ehs_bins = n_ehs_bins
        self.n_rollouts_train = n_rollouts_train
        self.n_rollouts_eval = n_rollouts_eval
        self.eval_dl = eval_dl
        self.k = k
        self.history = []
        self.emulator = None

        self.Q, self.N = init_Q_and_N(n_ehs_bins)
        if from_config:
            self.Q, self.N = self.load(".")

        # logging 
        self.round_results = []
    
    def load(self, dir: str):
        self.Q, self.N = load_Q_and_N(dir)
    
    def save(self, dir: str):
        save_Q_and_N(self.Q, self.N, dir)

    
    def set_emulator(self, player_num, max_round, small_blind_amount, ante_amount, blind_structure):
        self.emulator = Emulator()
        self.emulator.set_game_rule(player_num, max_round, small_blind_amount, ante_amount)
        self.emulator.set_blind_structure(blind_structure)
        # for info in players_info: # don't think i actually need this, since I never call run until finish
        #     self.emulator.register_player(info["uuid"], RandomPlayer())
    
    # TODO: what are we doing
    def train(self, n_games, players_info, save_dir: str):
        if self.emulator is None:
            raise ValueError("Emulator not set")

        initial_state = self.emulator.generate_initial_game_state(players_info)
        for i in tqdm(range(n_games), desc="Training games"):
            s = self.emulator.start_new_round(initial_state)
            init_stacks = _get_init_stacks(s)
            search(self.emulator, s, self.Q, self.N, self.n_rollouts_train, self.n_ehs_bins, init_stacks, float("inf"), k=self.k)
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
        s = (game_state, [])
        init_stacks = _get_init_stacks(s)
        # simulate for alloted budget 
        search(self.emulator, s, self.Q, self.N, self.n_rollouts_eval, self.n_ehs_bins, init_stacks, self.eval_dl, k=self.k)
        # argmax best valid action
        obs = get_obs(hole_card, round_state, self.uuid, self.n_ehs_bins)
        a_set = get_a_set(_get_valid_actions(s))
        a = select_action(self.Q, self.N, obs, a_set, c=0, use_ucb=False)
        return REAL_ACTIONS[a]

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.init_stack = _get_stack(seats, self.uuid)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        reward = compute_reward(winners, round_state, self.uuid, self.init_stack, USE_STACK_DIFF)
        
        # update Q function, N function
        self._log_result(winners, hand_info, round_state, reward)
    
    def _log_result(self, winners, hand_info, round_state, reward):
        round_log = get_round_log(round_state, reward, self.uuid)
        self.round_results.append(round_log)

    def setup_ai():
        return MCTSPlayer()