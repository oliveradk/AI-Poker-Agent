from pypokerengine.players import BasePokerPlayer 
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck

from q_learn_player import (
    REAL_ACTIONS,
    load_Q_and_N,
    get_obs,
    select_action,
    get_a_set, 
)
from mcts_player import search, _get_init_stacks, _get_valid_actions

def create_emulator(player_num, small_blind_amount, ante_amount, blind_structure):
    emulator = Emulator()
    emulator.set_game_rule(player_num, float("inf"), small_blind_amount, ante_amount)
    emulator.set_blind_structure(blind_structure)
    return emulator


class CustomPlayer(BasePokerPlayer):
    N_EHS_BINS = 5 
    N_ROLLOUTS = 2 
    EVAL_DL = 2
    K = 0.5
    
    def _init(self, round_state): 
        # load Q and N from file
        self.Q, self.N = load_Q_and_N(".")
        # set emulator
        player_num = len(round_state["seats"])
        small_blind_amount = round_state["small_blind_amount"]
        ante_amount = 0 
        blind_structure = {}
        self._set_emulator(player_num, small_blind_amount, ante_amount, blind_structure)

    
    def _set_emulator(self, player_num, small_blind_amount, ante_amount, blind_structure):
        self.emulator = create_emulator(player_num, small_blind_amount, ante_amount, blind_structure)

    def declare_action(self, valid_actions, hole_card, round_state):
        if not hasattr(self, "emulator"):
            self._init(round_state)
        if len(valid_actions) == 1: # not a real action if no choice
           return valid_actions[0]["action"]
        # get game state from round state 
        game_state = restore_game_state(round_state)
        for player in game_state["table"].seats.players:
           game_state = attach_hole_card_from_deck(game_state, player.uuid)
        s = (game_state, [])
        init_stacks = _get_init_stacks(s)
        # simulate for alloted budget 
        search(self.emulator, s, self.Q, self.N, self.N_ROLLOUTS, self.N_EHS_BINS, init_stacks, self.EVAL_DL, k=self.K)
        # argmax best valid action
        obs = get_obs(hole_card, round_state, self.uuid, self.N_EHS_BINS)
        a_set = get_a_set(_get_valid_actions(s))
        a = select_action(self.Q, self.N, obs, a_set, c=0, use_ucb=False)
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
        pass

    def setup_ai():
        return CustomPlayer()