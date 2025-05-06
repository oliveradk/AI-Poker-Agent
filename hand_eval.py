from deuces import Evaluator, Card, Deck
import random
EVALUATOR = Evaluator()


def str_to_card(card_str: str) -> Card:
    return Card.new(f"{card_str[1]}{card_str[0].lower()}")

def evaluate_hand(private_cards: list[str], community_cards: list[str], n_samples: int=1000) -> float:
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
        if our_hand_rank < opponent_hand_rank:
            wins += 1
        elif our_hand_rank == opponent_hand_rank:
            ties += 1
    ehs = (wins + 0.5 * ties) / n_samples
    return ehs

