from pypokerengine.players import BasePokerPlayer

# TODO: 
# 1. abstraction over actions (i.e. raise sizes)
# concretely: we have discrete set of actions: [hold, call, 2x, 3x, all in]
# a function of bet sizes, pot size, player stacks

# 2. Evaluation function
# a) preflop evaluation function 
# - fixed loopup table using position, hand strength, and player type (loose, normal, tight)
# b) learned post-flop evaluation function 
# - infered player type, hand strength, pot size, player stacks, position

# 3. MCTS search (used for post-flop evaluation)
# - for learning evaluation function, do actual rollouts with randomly sampled types 
# - for inference against real opponents, sample type based on prior to do rollouts

# 4. Card abstraction
# # "potential aware abstraction"...

# Information abstraction
# idk...


# First pass protoype: 
# get hard-coded preflop eval / ranges for each player type
# get basic action abstraction working
# get basic MCTS working

# TODO: look at https://cdn.aaai.org/ocs/ws/ws1227/8811-38072-1-PB.pdf

# refinements: 
# information sets (i.e. refine search based on possible hands of opponent)




# 2. abstraction over game state
# # suits (obviously)
# # preflop hand strenghts 

# concretely: 

# evaluation function: 
# 2 + 2 algorithm
# how does evaluation work in 

# maybe after community cards are revealed we just do (abstracted?) hand search



# # maybe combined hand strength eventually? 
# 3. opponent modeling
# # assume 3 types: conservative, normal, aggressive, with different hand ranges
# # assume uniform prior over types, if we observe them play a hand outside a given range, update prior 


# 4. Search over actions
# # monte carlo tree search with learned evaluation function
# # Learning / Training
# # Do MCTS using two copies randomly initializd with different types
# # fit linear model on abstracted game state (i.e, infered player type, hand strength, pot size, player stacks, position)
# # 


class OurPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        return "raise"
    
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