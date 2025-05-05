from pypokerengine.players import BasePokerPlayer
import numpy as np
import random
import deuces
from deuces import Deck, Card

REAL_ACTIONS = ["fold", "call", "raise"]
N_ACTIONS = len(REAL_ACTIONS)

EVALUATOR = deuces.Evaluator()


def _get_action_bin(action: dict | None):
    if action is None:
        return -1 
    action = action["action"]
    if action not in REAL_ACTIONS:
        return -1 
    return REAL_ACTIONS.index(action)

def str_to_card(card_str: str) -> Card:
    return Card.new(f"{card_str[1]}{card_str[0].lower()}")

# TODO: use hand evaluator
def get_EHS(private_cards, community_cards, n_bins: int, n_samples: int=1000):
    """
    Implements expected hand strength using monte-carlo simulation
    """
    # get set of remaining possible cards
    private_cards = [str_to_card(card) for card in private_cards]
    community_cards = [str_to_card(card) for card in community_cards]
    known_cards = set(private_cards + community_cards)
    deck = Deck().cards
    remaining_cards = [card for card in deck if card not in known_cards]
    # sample n_samples from this set of remaning community cards and opponent hards
    wins = 0 
    ties = 0
    community_needed = 5 - len(community_cards)
    opponent_needed = 2
    for _ in range(n_samples):
        sampled_cards = random.sample(remaining_cards, community_needed + opponent_needed)
        opponent_cards = sampled_cards[:opponent_needed]
        board = community_cards + sampled_cards[opponent_needed:]
        our_hand_rank = EVALUATOR.evaluate(private_cards, board)
        opponent_hand_rank = EVALUATOR.evaluate(opponent_cards, board)
        if our_hand_rank > opponent_hand_rank:
            wins += 1
        elif our_hand_rank == opponent_hand_rank:
            ties += 1
    ehs = (wins + 0.5 * ties) / n_samples
    ehs = ehs ** 2
    # bin value 
    ehs_bin = int(ehs * n_bins)
    if ehs_bin == n_bins:
        return n_bins - 1
    return ehs_bin

def get_obs(hole_card, round_state, n_bins):
    ehs_bin = get_EHS(hole_card, round_state["community_card"], n_bins)
    oppo_last_act_idx = _get_action_bin(_get_last_action(round_state))
    return oppo_last_act_idx, ehs_bin

# NOTE: this is technically variable for different actions...
def get_exploration_value(round_state, uuid, n_rounds: int=4, n_raises: int=3, k=0.5):
    pot_size = round_state["pot"]["main"]["amount"] # ignore side pots
    # remaining value this round
    n_bets = n_raises + 1
    oppo_bets = 0
    for action in round_state["action_histories"][round_state["street"]]:
        if action["uuid"] != uuid:
            oppo_bets += 1
    round_value = (n_bets - oppo_bets) * round_state["small_blind_amount"]
    # remaining value in future rounds
    max_rounds = n_rounds - round_state["round_count"]
    max_raise = n_bets * round_state["small_blind_amount"]
    future_round_value = max_rounds * max_raise
    return pot_size + k * (round_value + future_round_value)

def update(Q, N, oppo_last_act_idx, ehs_bin, action_idx, reward):
    N[oppo_last_act_idx, ehs_bin, action_idx] += 1
    N_s = N[oppo_last_act_idx, ehs_bin, action_idx]
    Q_s = Q[oppo_last_act_idx, ehs_bin, action_idx]
    update = (reward - Q_s) / N_s
    Q[oppo_last_act_idx, ehs_bin, action_idx] += update
    

def compute_reward(winners, round_state, uuid):
    # get player bets throughout round 
    player_bets = 0
    for street_bets in round_state["action_histories"].values():
        for bet in street_bets:
            if bet["uuid"] == uuid:
                if "amount" not in bet:
                    print(bet)
                player_bets += bet.get("paid", bet.get("amount", 0))
    # get pot size
    pot_size = round_state["pot"]["main"]["amount"]
    for side_pot in round_state["pot"]["side"]:
        pot_size += side_pot["amount"]
    # compute pot reward
    if len(winners) == 1 and winners[0]["uuid"] == uuid:
        pot_reward = pot_size 
    elif len(winners) == 2:
        pot_reward = pot_size / 2
    else:
        pot_reward = 0
    return pot_reward - player_bets

def get_a_set(valid_actions):
    valid_acts = [action["action"] for action in valid_actions]
    return [i for i, action in enumerate(REAL_ACTIONS) if action in valid_acts]


def select_action(Q, N, obs, a_set, c):
    ucb = c * np.sqrt(
        np.log(N[obs[0], obs[1], :].sum()) / (N[obs[0], obs[1], :])
    )
    vals = Q[obs[0], obs[1], :] + ucb
    vals[np.array(a_set)] = float("-inf")
    return np.argmax(vals)

 # TODO: implement "smooth" ucb
def sample_action(
    Q: np.ndarray, 
    N: np.ndarray, 
    obs: tuple[int, int],
    valid_actions: list[dict], 
    c: float=1.4
):
    a_set = get_a_set(valid_actions)
    if N[obs[0], obs[1], :].sum() == 0:
        action_idx = np.random.choice(a_set)
    else:
        action_idx = select_action(Q, N, obs, a_set, c)
    return action_idx

def _get_last_action(round_state):
    round_actions = round_state["action_histories"][round_state["street"]]
    if len(round_actions) == 0:
        return None
    return round_actions[-1]
    

class QLearningPlayer(BasePokerPlayer):
    # based on https://cdn.aaai.org/ocs/ws/ws1227/8811-38072-1-PB.pdf
    # NOTE: currently the agent is unaware of rounds
    
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
    
    def declare_action(self, valid_actions, hole_card, round_state):
       if len(valid_actions) == 1: # not a real action if no choice
           return valid_actions[0]["action"]
       # get state
       obs = get_obs(hole_card, round_state, self.n_ehs_bins)
       # compute exploration value from max payoff 
       c = get_exploration_value(round_state, self.uuid, n_rounds=4, n_raises=3, k=self.k)
       # sample action
       action_idx = sample_action(self.Q, self.N, obs, valid_actions, c=c)
       # update history
       if self.is_training:
           self.history.append(obs + (action_idx,))
       return REAL_ACTIONS[action_idx]
    

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        # update current stack
        self.cur_stack = self._my_stack(seats)
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # compute reward 
        reward = compute_reward(winners, round_state, self.uuid)
        # update Q function, N function
        for (oppo_last_act_idx, ehs_bin, action_idx) in self.history:
            update(self.Q, self.N, oppo_last_act_idx, ehs_bin, action_idx, reward)
        # restart history
        self.history = []
        # log result
        self._log_result(winners, hand_info, round_state, reward)

    
    def _log_result(self, winners, hand_info, round_state, reward):
        # Calculate pot size
        pot_size = round_state["pot"]["main"]["amount"]
        for side_pot in round_state["pot"]["side"]:
            pot_size += side_pot["amount"]
        seat = [s for s in round_state["seats"] if s["uuid"] == self.uuid][0]
        stack = seat["stack"]
        
        # Log result
        result = {
            "round": round_state["round_count"],
            "reward": reward,
            "pot_size": pot_size,
            "street": round_state["street"],
            "community_cards": round_state["community_card"],
            "stack": stack,
        }
        self.round_results.append(result)


    def _my_stack(self, seats):
        my_seat = [s for s in seats if s["uuid"] == self.uuid][0]
        return my_seat["stack"]

    
    def _get_action_bin(self, action: dict | None):
        if action is None:
            return -1 
        action = action["action"]
        if action not in REAL_ACTIONS:
            return -1 
        return REAL_ACTIONS.index(action)

if __name__ == "__main__":
    player = QLearningPlayer(n_ehs_bins=10, is_training=True)
    player.load_Q(Q_path="q_values.npy")
