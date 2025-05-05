from pypokerengine.players import BasePokerPlayer
import numpy as np
import random
import deuces
from deuces import Deck, Card

REAL_ACTIONS = ["fold", "call", "raise"]
N_ACTIONS = len(REAL_ACTIONS)

# TODO: implement actual MCTS

def str_to_card(card_str: str) -> Card:
    return Card.new(f"{card_str[1]}{card_str[0].lower()}")

def get_EHS(private_cards, community_cards, n_bins: int, n_samples: int=1000, evaluator: deuces.Evaluator | None = None):
    """
    Implements expected hand strength using monte-carlo simulation
    """
    if evaluator is None:
        evaluator = deuces.Evaluator()
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
        our_hand_rank = evaluator.evaluate(private_cards, board)
        opponent_hand_rank = evaluator.evaluate(opponent_cards, board)
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

 # TODO: implement "smooth" ucb
def sample_action(
    Q: np.ndarray, 
    N: np.ndarray, 
    oppo_last_action: int, 
    ehs_bin: int, 
    valid_actions: list[dict], 
    use_ucb: bool=True, 
    c: float=1.4
):
    # get valid action indices
    valid_actions = [action["action"] for action in valid_actions]
    valid_action_mask = np.array([action in valid_actions for action in REAL_ACTIONS], dtype=float)
    # compute ucb 
    ucb = np.zeros(N_ACTIONS)
    if use_ucb:
        if N[oppo_last_action, ehs_bin, :].sum() == 0:
            return np.random.choice(N_ACTIONS)
        ucb += c * np.sqrt(
            np.log(N[oppo_last_action, ehs_bin, :].sum()) / (N[oppo_last_action, ehs_bin, :])
        )
    # sample action 
    vals = Q[oppo_last_action, ehs_bin, :] + ucb
    return np.argmax(vals * valid_action_mask)

def _get_last_action(round_state):
    round_actions = round_state["action_histories"][round_state["street"]]
    if len(round_actions) == 0:
        return None
    return round_actions[-1]
    


# so lets go through basic q-learning

# start game (initial hands dealt)
# compute EMS (using private cards)
# sample action from value function (can use smooth thing)
# add actions to current history
# when terminal state reached, update value function:
# for each action in history, update the mean value using the reward



class MCTSPlayer(BasePokerPlayer):
    # based on https://cdn.aaai.org/ocs/ws/ws1227/8811-38072-1-PB.pdf
    # NOTE: currently the agent is unaware of rounds
    
    def __init__(self, n_ehs_bins: int, is_training: bool, k: float=0.5, use_stack_diff: bool=True): 
        self.n_ehs_bins = n_ehs_bins
        # Q[opponent_last_action, hand_strength_bin, action_idx]
        # Q[-1, :, :] is reserved for when you go first
        self.is_training = is_training
        self.Q = np.zeros((N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.N = np.zeros((N_ACTIONS+1, n_ehs_bins, N_ACTIONS))
        self.k = k
        self.history = []
        self.evaluator = deuces.Evaluator()
        
        # logging 
        self.round_results = []

        # computing reward
        self.use_stack_diff = use_stack_diff
        self.cur_stack = 0
        
    
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
       # get EM
       ehs_bin = get_EHS(hole_card, round_state["community_card"], self.n_ehs_bins, evaluator=self.evaluator)
       # get last action
       oppo_last_act_idx = self._get_action_bin(_get_last_action(round_state))
       # compute exploration value from max payoff 
       c = get_exploration_value(round_state, self.uuid, n_rounds=4, n_raises=3, k=self.k) if self.use_stack_diff else np.sqrt(2)
       # sample action
       action_idx = sample_action(self.Q, self.N, oppo_last_act_idx, ehs_bin, valid_actions, c=c)
       # update history
       if self.is_training:
           self.history.append((oppo_last_act_idx, ehs_bin, action_idx))
       return REAL_ACTIONS[action_idx]
    

    # TODO: update c too
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
        reward = self._compute_reward(winners, hand_info, round_state, self.use_stack_diff)
        # update Q function, N function
        for (oppo_last_act_idx, ehs_bin, action_idx) in self.history:
            self._update(oppo_last_act_idx, ehs_bin, action_idx, reward)
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
    
    def _update(self, oppo_last_act_idx, ehs_bin, action_idx, reward):
        self.N[oppo_last_act_idx, ehs_bin, action_idx] += 1
        N_s = self.N[oppo_last_act_idx, ehs_bin, action_idx]
        Q_s = self.Q[oppo_last_act_idx, ehs_bin, action_idx]
        update = (reward - Q_s) / N_s
        self.Q[oppo_last_act_idx, ehs_bin, action_idx] += update

    def _compute_reward(self, winners, hand_info, round_state, use_stack_diff: bool=True):
        # win loss reward
        # TODO: implement pot weighed reward, expected value of winnings, etc.
        won = any([winner["uuid"] == self.uuid for winner in winners])
        if use_stack_diff:
            new_stack = self._my_stack(round_state['seats'])
            if won: 
                new_stack += round_state["pot"]["main"]["amount"]
            reward = new_stack - self.cur_stack
        else: 
            reward = 1.0 / len(winners) if won else 0.0
        return reward
    
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
    player = MCTSPlayer(n_ehs_bins=10, is_training=True)
    player.load_Q(Q_path="q_values.npy")
