from pypokerengine.players import BasePokerPlayer
import numpy as np
import os
from hand_eval import evaluate_hand

REAL_ACTIONS = ["fold", "call", "raise"]
STREET_IDX = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
}
N_ACTIONS = len(REAL_ACTIONS)
N_STREETS = 4
N_RAISES = 4 
SMALL_BET = 2 
BIG_BET = 4
MAX_GAIN = 2 * (N_RAISES) * SMALL_BET + 2 * (N_RAISES) * BIG_BET


def _get_action_bin(action: dict | None):
    if action is None:
        return -1 
    action = action["action"]
    if action not in REAL_ACTIONS:
        return -1 
    return REAL_ACTIONS.index(action)

def _get_dealer_pos(round_state):
    return round_state["dealer_btn"]

def _get_stack(seats, uuid):
    seat = next((s for s in seats if s["uuid"] == uuid))
    return seat["stack"]

def _get_last_aggressor(round_state, uuid):
    street = round_state["street"]
    for action in reversed(round_state["action_histories"][street]):
        if action["action"] == "RAISE":
                return int(action["uuid"] != uuid)
    return 0 # default to 0 if no last aggressor - agent can tell b/c raise count is 0

def _get_raise_count(round_state):
    raise_count = 0
    street = round_state["street"]
    for action in round_state["action_histories"][street]:
        if action["action"] == "RAISE":
            raise_count += 1
    return raise_count


def init_Q_and_N(n_bins: int):
    # Q[position, street, street raises, last_aggressor, hand_strength_bin, action_idx] (5760)
    Q = np.zeros((2, N_STREETS, 1+N_RAISES, 2, n_bins, N_ACTIONS)) #2 x 4 x 5 x 5 
    N = np.zeros((2, N_STREETS, 1+N_RAISES, 2, n_bins, N_ACTIONS))
    return Q, N

def load_Q_and_N(dir: str):
    Q = np.load(os.path.join(dir, "Q.npy"))
    N = np.load(os.path.join(dir, "N.npy"))
    return Q, N

def save_Q_and_N(Q, N, dir: str):
    np.save(os.path.join(dir, "Q.npy"), Q)
    np.save(os.path.join(dir, "N.npy"), N)


def get_EHS(private_cards, community_cards, n_bins: int, n_samples: int=1000):
    """
    Implements expected hand strength using monte-carlo simulation
    """
    ehs = evaluate_hand(private_cards, community_cards, n_samples)
    ehs = ehs ** 2
    # bin value 
    ehs_bin = int(ehs * n_bins)
    if ehs_bin == n_bins:
        return n_bins - 1
    return ehs_bin

def get_obs(hole_card, round_state, uuid, n_bins):
    # position, street, last_action, ehs_bin
    position = 0 if round_state["seats"][_get_dealer_pos(round_state)]["uuid"] == uuid else 1
    raises = _get_raise_count(round_state)
    last_aggressor = _get_last_aggressor(round_state, uuid)
    street = STREET_IDX[round_state["street"]]
    ehs_bin = get_EHS(hole_card, round_state["community_card"], n_bins)
    return position, street, raises, last_aggressor, ehs_bin

# TODO: add the round-weighting thing
def get_explore_weight(round_state, uuid, use_stack_diff, k=0.5):
    if not use_stack_diff:
        return np.sqrt(2.0)
    # for now just do maximum gain 
    return MAX_GAIN * k

def update(Q, N, obs_a, reward):
    N[obs_a] += 1
    N_s = N[obs_a]
    Q_s = Q[obs_a]
    update = (reward - Q_s) / N_s
    Q[obs_a] += update
    

def compute_reward(winners, round_state, uuid, init_stack, use_stack_diff):
    if len(winners) == 1 and winners[0]["uuid"] == uuid:
        frac_pot = 1.0
    elif len(winners) == 2:
        frac_pot = 0.5
    else:
        frac_pot = 0.0
    if not use_stack_diff: 
        return frac_pot
    pot = round_state["pot"]["main"]["amount"]
    cur_stack = _get_stack(round_state["seats"], uuid)
    chip_diff = cur_stack - init_stack + pot * frac_pot
    sb_diff = chip_diff / round_state["small_blind_amount"]
    return sb_diff

def get_a_set(valid_actions):
    valid_acts = [action["action"] for action in valid_actions]
    return [i for i, action in enumerate(REAL_ACTIONS) if action in valid_acts]


def select_action(Q, N, obs, a_set, c, use_ucb: bool=True):
    # TODO: add option for smooth ucb
    Q_obs = Q[obs].copy()
    if use_ucb:
        ucb = c * np.sqrt(
            np.log(N[obs].sum()) / (N[obs])
        )
        Q_obs += ucb
    vals = -np.inf * np.ones(N_ACTIONS)
    vals[a_set] = Q_obs[a_set]
    return np.argmax(vals)

 # TODO: implement "smooth" ucb
def sample_action(
    Q: np.ndarray, 
    N: np.ndarray, 
    obs: tuple[int, ...],
    valid_actions: list[dict], 
    c: float=1.4
):
    a_set = get_a_set(valid_actions)
    if N[obs].sum() == 0:
        action_idx = np.random.choice(a_set)
    else:
        action_idx = select_action(Q, N, obs, a_set, c)
    return action_idx

def _get_last_action(round_state):
    round_actions = round_state["action_histories"][round_state["street"]]
    if len(round_actions) == 0:
        return None
    return round_actions[-1]

def get_round_log(round_state, reward, uuid):
    # Calculate pot size
    pot_size = round_state["pot"]["main"]["amount"]
    for side_pot in round_state["pot"]["side"]:
        pot_size += side_pot["amount"]
    seat = [s for s in round_state["seats"] if s["uuid"] == uuid][0]
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
    return result

    

class QLearningPlayer(BasePokerPlayer):
    # based on https://cdn.aaai.org/ocs/ws/ws1227/8811-38072-1-PB.pdf
    # NOTE: currently the agent is unaware of rounds
    
    def __init__(self, n_ehs_bins: int, is_training: bool, k: float=0.5, use_stack_diff: bool=False): 
        self.n_ehs_bins = n_ehs_bins
        self.is_training = is_training
        self.Q, self.N = init_Q_and_N(n_ehs_bins)
        self.k = k
        self.history = []
        self.use_stack_diff = use_stack_diff
        
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
       obs = get_obs(hole_card, round_state, self.uuid, self.n_ehs_bins)
       # compute exploration value from max payoff 
       c = get_explore_weight(round_state, self.uuid, self.use_stack_diff, k=self.k)
       # sample action
       action_idx = sample_action(self.Q, self.N, obs, valid_actions, c=c)
       # update history
       if self.is_training:
           self.history.append(obs + (action_idx,))
       return REAL_ACTIONS[action_idx]
    

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.init_stack = _get_stack(seats, self.uuid)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # compute reward 
        reward = compute_reward(winners, round_state, self.uuid, self.init_stack, self.use_stack_diff)
        # update Q function, N function
        for obs_a in self.history:
            update(self.Q, self.N, obs_a, reward)
        # restart history
        self.history = []
        # log result
        self._log_result(winners, hand_info, round_state, reward)

        round_number = round_state["round_count"]
        if round_number % 100 == 0:
            print(f"Round {round_number} complete")
            print("frac states explored", np.mean(self.N > 0))
            print("mean action dist", np.mean(self.Q, axis=tuple(range(len(self.Q.shape) - 1))))

    
    def _log_result(self, winners, hand_info, round_state, reward):
        round_log = get_round_log(round_state, reward, self.uuid)
        self.round_results.append(round_log)

    
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
