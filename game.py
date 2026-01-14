from agent import Agent
from typing import List, Dict
import random


class TheMindGame:
    """
    The Mind card game implementation.

    Rules:
    - N players, each with K cards from range [1, 100]
    - Must play all cards in ascending order (globally)
    - No communication - only observe when others play
    - Lives system: start with L lives, lose 1 per mistake
    - Game ends when all cards played OR lives = 0
    """

    def __init__(self, cards_per_player: int = 2, max_card: int = 100):
        self.cards_per_player = cards_per_player
        self.max_card = max_card

    def play_game(self, agents: List[Agent], max_ticks: int = 3000,
                  verbose: bool = False) -> Dict:
        """Play one complete game."""

        # Deal cards
        total_cards = self.cards_per_player * len(agents)
        all_cards = random.sample(range(1, self.max_card + 1),
                                  total_cards)
        random.shuffle(all_cards)

        for i, agent in enumerate(agents):
            agent_cards = all_cards[i * self.cards_per_player:(
                                                                          i + 1) * self.cards_per_player]
            agent.receive_cards(agent_cards)

        if verbose:
            for a in agents:
                print(
                    f"  Agent {a.agent_id} hand: {sorted(a.hand)} (wf={a.wait_factor:.2f})")

        # Game state
        cards_played_correctly = 0
        mistakes = 0
        lives = self.cards_per_player - 1  # More lives with more cards
        play_log = []

        for tick in range(max_ticks):
            # Check termination
            if all(len(a.hand) == 0 for a in agents):
                break  # Won!
            if lives <= 0:
                break  # Lost!

            # Collect decisions
            decisions = {}
            for agent in agents:
                should_play, card = agent.decide(tick)
                if should_play and card > 0:
                    decisions[agent.agent_id] = card

            if not decisions:
                continue

            # Process plays (random order for ties)
            plays_this_tick = list(decisions.items())
            random.shuffle(plays_this_tick)

            for agent_id, card in plays_this_tick:
                if lives <= 0:
                    break

                agent = agents[agent_id]
                other = agents[1 - agent_id]

                # Find global lowest unplayed card
                lowest_unplayed = float('inf')
                for a in agents:
                    if a.hand:
                        lowest_unplayed = min(lowest_unplayed,
                                              min(a.hand))

                if card == lowest_unplayed:
                    # Correct play
                    agent.play_card(card)
                    cards_played_correctly += 1

                    # Partner observes (ToM)
                    other.observe_partner_play(card, tick)

                    play_log.append({'tick': tick, 'agent': agent_id,
                                     'card': card, 'correct': True})
                    if verbose:
                        print(
                            f"  Tick {tick:4d}: Agent {agent_id} plays {card:2d} ✓")
                else:
                    # Mistake
                    mistakes += 1
                    lives -= 1
                    agent.play_card(card)

                    play_log.append({'tick': tick, 'agent': agent_id,
                                     'card': card, 'correct': False})
                    if verbose:
                        print(
                            f"  Tick {tick:4d}: Agent {agent_id} plays {card:2d} ✗ "
                            f"(should wait for {lowest_unplayed}) Lives: {lives}")

        return {
            'cards_played': cards_played_correctly,
            'total_cards': total_cards,
            'mistakes': mistakes,
            'lives_left': lives,
            'success_rate': cards_played_correctly / total_cards,
            'mistake_rate': mistakes / total_cards,
            'won': lives > 0 and all(len(a.hand) == 0 for a in agents),
            'play_log': play_log
        }

