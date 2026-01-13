"""
===============================================================================
THE MIND - Common Ground Simulation
Modeling Theory of Mind and Common Ground in Agent-Agent Interaction
===============================================================================

Based on: Van der Meulen, Verbrugge, & Van Duijn (2024)
"Common Ground Provides a Mental Shortcut in Agent-Agent Interaction"

HYPOTHESES (from paper):
  H1: Accounting for the other's perspective using ToM increases performance
  H2: Establishing CG retains performance while decreasing active modeling

KEY CONCEPTS:
  - Theory of Mind (ToM): Actively inferring partner's strategy from observations
  - Common Ground (CG): Shared conventions that eliminate need for active inference
  - "Mental Shortcut": Once CG established, agents stop updating their partner model

EXPERIMENTAL DESIGN:
  - Condition A: Agents WITH Theory of Mind (observe and model partner)
  - Condition B: Agents WITHOUT ToM (egocentric, no partner modeling)
  - Compare: Performance, CG establishment rate, modeling effort over time

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random
from scipy import stats
import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class Agent:
    """
    Agent that plays The Mind using timing-based coordination.
    
    Core Parameter:
        wait_factor: Determines timing as wait_time = card_value * wait_factor
        Higher = more patient, lower = more impulsive
    
    Theory of Mind Implementation (Simulation-ToM, per Gallese & Goldman 1998):
        - Agents implicitly predict partner behavior through their own reasoning
        - They observe partner's play timing and infer partner's wait_factor
        - They adjust their own behavior to complement partner
    
    Common Ground:
        - Established when both agents' strategies stabilize
        - Once CG detected, agent STOPS updating (mental shortcut)
        - This reduces "cognitive load" of active modeling
    """
    
    def __init__(self, agent_id: int, wait_factor: float = 1.0, 
                 learning_rate: float = 0.05, use_tom: bool = True):
        self.agent_id = agent_id
        self.wait_factor = np.clip(wait_factor, 0.3, 5.0)
        self.learning_rate = learning_rate
        self.use_tom = use_tom  # Toggle for experimental comparison
        
        self.initial_wait_factor = self.wait_factor
        
        # ===== Theory of Mind: Partner Model =====
        self.partner_wait_factor_estimate = 1.0  # Initial assumption
        self.partner_observations = deque(maxlen=100)  # (card, tick) pairs
        self.tom_updates_count = 0  # Track "cognitive effort"
        
        # ===== Learning History =====
        self.wait_factor_history = []
        self.partner_estimate_history = []
        self.score_history = deque(maxlen=30)
        self.mistake_history = deque(maxlen=30)
        
        # ===== Common Ground State =====
        self.cg_established = False
        self.cg_round = None
        self.cg_wait_factor = None  # Locked strategy after CG
        
        # ===== Current Game State =====
        self.hand: List[int] = []
        self.planned_play_times: Dict[int, float] = {}
        
    def receive_cards(self, cards: List[int]):
        """Receive cards and plan play times using Weber's Law timing."""
        self.hand = sorted(cards)
        self._plan_play_times()
        
    def _plan_play_times(self):
        """
        Generate human-like timing using Weber's Law.
        
        Weber's Law: The just-noticeable difference in a stimulus is 
        proportional to the magnitude of the stimulus.
        
        Applied here: Timing uncertainty scales with the target duration.
        For a card with value C and wait_factor W:
            target_time = C * W
            noise_std = target_time * weber_fraction
        """
        self.planned_play_times = {}
        weber_fraction = 0.15  # Empirically derived for timing tasks
        
        for card in self.hand:
            target_wait = card * self.wait_factor
            
            # Weber's Law: noise proportional to magnitude
            noise_std = max(0.5, target_wait * weber_fraction)
            actual_time = target_wait + np.random.normal(0, noise_std)
            
            self.planned_play_times[card] = max(0, actual_time)

    def decide(self, current_tick: int) -> Tuple[bool, int]:
        """Decide whether to play lowest card this tick."""
        if not self.hand:
            return False, -1
            
        my_lowest = min(self.hand)
        play_time = self.planned_play_times.get(my_lowest, float('inf'))
        
        if current_tick >= play_time:
            return True, my_lowest
        return False, -1
    
    def play_card(self, card: int):
        """Remove card from hand after playing."""
        if card in self.hand:
            self.hand.remove(card)
            
    def observe_partner_play(self, partner_card: int, tick_played: int):
        """
        Theory of Mind: Observe partner's play and update mental model.
        
        From observed (card, tick) pairs, infer partner's wait_factor:
            inferred_wf = tick_played / card_value
        
        This is ONLY called if self.use_tom == True
        """
        if not self.use_tom or self.cg_established:
            return
            
        if partner_card > 0:
            # Infer partner's wait_factor from their timing
            inferred_wf = tick_played / partner_card
            inferred_wf = np.clip(inferred_wf, 0.3, 5.0)
            
            self.partner_observations.append((partner_card, tick_played, inferred_wf))
            self.tom_updates_count += 1  # Track cognitive effort
            
            # Update partner model with exponential moving average
            if len(self.partner_observations) >= 3:
                alpha = 0.15  # Learning rate for partner model
                recent_wfs = [obs[2] for obs in list(self.partner_observations)[-10:]]
                recent_mean = np.mean(recent_wfs)
                
                self.partner_wait_factor_estimate = (
                    (1 - alpha) * self.partner_wait_factor_estimate +
                    alpha * recent_mean
                )
    
    def update_after_round(self, success_rate: float, mistake_rate: float, 
                          round_num: int):
        """
        Update strategy based on round outcomes.
        
        Learning Rules:
        1. MISTAKE → Increase wait_factor (be more patient)
        2. PERFECT GAME → Slightly decrease wait_factor (can be faster)
        3. ToM ALIGNMENT → Move toward partner's estimated timing
        """
        if self.cg_established:
            return  # Mental shortcut: no more updates
            
        self.score_history.append(success_rate)
        self.mistake_history.append(mistake_rate)
        self.wait_factor_history.append(self.wait_factor)
        self.partner_estimate_history.append(self.partner_wait_factor_estimate)
        
        # ----- Rule 1: Crash Avoidance (mistakes → slow down) -----
        if mistake_rate > 0:
            slowdown = self.learning_rate * (1.0 + mistake_rate * 3.0)
            self.wait_factor += slowdown
            
        # ----- Rule 2: Efficiency (perfect → speed up slightly) -----
        elif success_rate >= 0.95:
            speedup = self.learning_rate * 0.15
            self.wait_factor -= speedup
            
        # ----- Rule 3: ToM-based Alignment (ONLY if using ToM) -----
        if self.use_tom and len(self.partner_observations) >= 5:
            # Move toward partner's timing (convergence)
            partner_wf = self.partner_wait_factor_estimate
            diff = partner_wf - self.wait_factor
            
            # Stronger alignment when performing poorly
            alignment_strength = 0.4 if mistake_rate > 0.1 else 0.2
            self.wait_factor += diff * self.learning_rate * alignment_strength
        
        # Non-ToM agents: purely egocentric, no partner alignment
        # They only learn from their own outcomes
            
        # ----- Random exploration -----
        self.wait_factor += np.random.normal(0, 0.02)
        self.wait_factor = np.clip(self.wait_factor, 0.3, 5.0)
        
        # Check for Common Ground
        self._check_cg(round_num)
        
    def _check_cg(self, round_num: int):
        """
        Check if Common Ground is established.
        
        CG Definition (from paper):
        "Information two or more individuals have in common about a given 
        scenario, and access to the knowledge that they both have this information"
        
        Operationalized as:
        1. Own strategy (wait_factor) has stabilized
        2. Partner model has stabilized (if using ToM)
        3. Performance is consistently good
        """
        min_round = 20 + random.randint(0, 10)
        
        if round_num < min_round:
            return
        if len(self.wait_factor_history) < 20:
            return
            
        # Condition 1: Own strategy stability
        recent_wf = self.wait_factor_history[-15:]
        wf_stable = np.std(recent_wf) < 0.12
        
        # Condition 2: Partner model stability (only if using ToM)
        if self.use_tom and len(self.partner_estimate_history) >= 15:
            recent_partner = self.partner_estimate_history[-15:]
            partner_stable = np.std(recent_partner) < 0.15
        else:
            partner_stable = True  # Non-ToM agents skip this check
            
        # Condition 3: Performance stability
        if len(self.score_history) >= 15:
            recent_scores = list(self.score_history)[-15:]
            perf_stable = np.mean(recent_scores) > 0.80 and np.std(recent_scores) < 0.15
        else:
            perf_stable = False
            
        # All conditions must be met
        if wf_stable and partner_stable and perf_stable:
            self.cg_established = True
            self.cg_round = round_num
            self.cg_wait_factor = self.wait_factor
            
    def get_state(self) -> Dict:
        """Return current agent state for logging."""
        return {
            'wait_factor': self.wait_factor,
            'partner_estimate': self.partner_wait_factor_estimate,
            'tom_updates': self.tom_updates_count,
            'cg_established': self.cg_established,
            'cg_round': self.cg_round
        }


# =============================================================================
# GAME IMPLEMENTATION
# =============================================================================

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
        all_cards = random.sample(range(1, self.max_card + 1), total_cards)
        random.shuffle(all_cards)
        
        for i, agent in enumerate(agents):
            agent_cards = all_cards[i * self.cards_per_player:(i + 1) * self.cards_per_player]
            agent.receive_cards(agent_cards)
        
        if verbose:
            for a in agents:
                print(f"  Agent {a.agent_id} hand: {sorted(a.hand)} (wf={a.wait_factor:.2f})")
        
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
                        lowest_unplayed = min(lowest_unplayed, min(a.hand))
                
                if card == lowest_unplayed:
                    # Correct play
                    agent.play_card(card)
                    cards_played_correctly += 1
                    
                    # Partner observes (ToM)
                    other.observe_partner_play(card, tick)
                    
                    play_log.append({'tick': tick, 'agent': agent_id, 
                                    'card': card, 'correct': True})
                    if verbose:
                        print(f"  Tick {tick:4d}: Agent {agent_id} plays {card:2d} ✓")
                else:
                    # Mistake
                    mistakes += 1
                    lives -= 1
                    agent.play_card(card)
                    
                    play_log.append({'tick': tick, 'agent': agent_id, 
                                    'card': card, 'correct': False})
                    if verbose:
                        print(f"  Tick {tick:4d}: Agent {agent_id} plays {card:2d} ✗ "
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


# =============================================================================
# EXPERIMENT
# =============================================================================

class Experiment:
    """
    Experimental framework for testing hypotheses.
    
    H1: ToM agents outperform non-ToM agents
    H2: After CG, ToM effort decreases without performance loss
    """
    
    def __init__(self, num_rounds: int = 80, games_per_round: int = 30):
        self.num_rounds = num_rounds
        self.games_per_round = games_per_round
        self.game = TheMindGame(cards_per_player=4)  # Harder task
        
    def create_agents(self, condition: str, use_tom: bool = True) -> List[Agent]:
        """Create agent pair with specified initial conditions."""
        configs = {
            'similar': {
                0: {'wait_factor': 1.1},
                1: {'wait_factor': 0.9}
            },
            'different': {
                0: {'wait_factor': 1.8},
                1: {'wait_factor': 0.5}
            },
            'very_different': {
                0: {'wait_factor': 3.0},
                1: {'wait_factor': 0.4}
            }
        }
        
        cfg = configs.get(condition, configs['different'])
        return [
            Agent(0, wait_factor=cfg[0]['wait_factor'], 
                  learning_rate=0.05, use_tom=use_tom),
            Agent(1, wait_factor=cfg[1]['wait_factor'], 
                  learning_rate=0.05, use_tom=use_tom)
        ]
    
    def run_condition(self, condition: str, use_tom: bool, seed: int) -> Dict:
        """Run one experimental condition."""
        random.seed(seed)
        np.random.seed(seed)
        
        agents = self.create_agents(condition, use_tom)
        tom_label = "ToM" if use_tom else "NoToM"
        
        results = {
            'condition': condition,
            'use_tom': use_tom,
            'seed': seed,
            'round_scores': [],
            'round_mistakes': [],
            'round_wins': [],
            'tom_effort': [],  # Track cognitive load
            'agent_states': {0: [], 1: []},
            'cg_rounds': {}
        }
        
        for round_num in range(self.num_rounds):
            round_scores = []
            round_mistakes = []
            round_wins = []
            
            for _ in range(self.games_per_round):
                result = self.game.play_game(agents)
                round_scores.append(result['success_rate'])
                round_mistakes.append(result['mistake_rate'])
                round_wins.append(1 if result['won'] else 0)
                
            avg_score = np.mean(round_scores)
            avg_mistakes = np.mean(round_mistakes)
            win_rate = np.mean(round_wins)
            
            results['round_scores'].append(avg_score)
            results['round_mistakes'].append(avg_mistakes)
            results['round_wins'].append(win_rate)
            
            # Track ToM effort (updates this round)
            tom_effort_this_round = sum(a.tom_updates_count for a in agents)
            results['tom_effort'].append(tom_effort_this_round)
            
            # Update agents
            for agent in agents:
                agent.update_after_round(avg_score, avg_mistakes, round_num)
                results['agent_states'][agent.agent_id].append(agent.get_state().copy())
                
                if agent.cg_established and agent.agent_id not in results['cg_rounds']:
                    results['cg_rounds'][agent.agent_id] = round_num
                    
        return results


def run_full_experiment(n_runs: int = 20):
    """
    Run complete experiment comparing ToM vs non-ToM agents.
    """
    print("="*70)
    print("THE MIND - COMMON GROUND EXPERIMENT")
    print("Testing H1 (ToM improves performance) and H2 (CG reduces effort)")
    print("="*70)
    
    all_results = {
        'tom': {'similar': [], 'different': [], 'very_different': []},
        'no_tom': {'similar': [], 'different': [], 'very_different': []}
    }
    
    exp = Experiment(num_rounds=80, games_per_round=30)
    
    for run in range(n_runs):
        if (run + 1) % 5 == 0:
            print(f"Run {run + 1}/{n_runs}...")
            
        for condition in ['similar', 'different', 'very_different']:
            base_seed = 10000 + run * 1000 + hash(condition) % 500
            
            # ToM condition
            result_tom = exp.run_condition(condition, use_tom=True, seed=base_seed)
            all_results['tom'][condition].append(result_tom)
            
            # Non-ToM condition (same seed for fair comparison)
            result_no_tom = exp.run_condition(condition, use_tom=False, seed=base_seed)
            all_results['no_tom'][condition].append(result_no_tom)
            
    return all_results


# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================

def analyze_hypotheses(results: Dict):
    """
    Statistical analysis of hypotheses.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING")
    print("="*70)
    
    # ===== H1: ToM improves performance =====
    print("\n--- H1: ToM improves performance ---")
    print("(Comparing final success rates)")
    
    for condition in ['similar', 'different', 'very_different']:
        tom_final = [r['round_scores'][-1] for r in results['tom'][condition]]
        no_tom_final = [r['round_scores'][-1] for r in results['no_tom'][condition]]
        
        t_stat, p_val = stats.ttest_ind(tom_final, no_tom_final)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(tom_final)**2 + np.std(no_tom_final)**2) / 2)
        cohens_d = (np.mean(tom_final) - np.mean(no_tom_final)) / pooled_std if pooled_std > 0 else 0
        
        print(f"\n{condition.upper()}:")
        print(f"  ToM final score:    {np.mean(tom_final):.3f} ± {np.std(tom_final):.3f}")
        print(f"  Non-ToM final:      {np.mean(no_tom_final):.3f} ± {np.std(no_tom_final):.3f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f} {'*' if p_val < 0.05 else ''}")
    
    # ===== H1 Alternative: ToM speeds up convergence =====
    print("\n--- H1 (Alternative): ToM accelerates learning ---")
    print("(Comparing time to Common Ground)")
    
    for condition in ['similar', 'different', 'very_different']:
        tom_cg = [max(r['cg_rounds'].values()) for r in results['tom'][condition] 
                  if len(r['cg_rounds']) == 2]
        no_tom_cg = [max(r['cg_rounds'].values()) for r in results['no_tom'][condition] 
                    if len(r['cg_rounds']) == 2]
        
        if tom_cg and no_tom_cg:
            t_stat, p_val = stats.ttest_ind(tom_cg, no_tom_cg)
            diff = np.mean(no_tom_cg) - np.mean(tom_cg)
            
            print(f"\n{condition.upper()}:")
            print(f"  ToM time to CG:     {np.mean(tom_cg):.1f} ± {np.std(tom_cg):.1f} rounds")
            print(f"  Non-ToM time to CG: {np.mean(no_tom_cg):.1f} ± {np.std(no_tom_cg):.1f} rounds")
            print(f"  ToM is {diff:.1f} rounds faster")
            print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
        
    # ===== H2: CG reduces modeling effort =====
    print("\n--- H2: CG reduces active modeling effort ---")
    
    for condition in ['similar', 'different', 'very_different']:
        # Compare ToM effort before vs after CG
        pre_cg_effort = []
        post_cg_effort = []
        
        for r in results['tom'][condition]:
            if len(r['cg_rounds']) == 2:
                cg_round = max(r['cg_rounds'].values())
                # Take derivative of cumulative effort (effort per round)
                effort = r['tom_effort']
                effort_per_round = np.diff([0] + effort)
                
                pre_cg_effort.append(np.mean(effort_per_round[:cg_round]))
                post_cg_effort.append(np.mean(effort_per_round[cg_round:]))
        
        if pre_cg_effort and post_cg_effort:
            t_stat, p_val = stats.ttest_rel(pre_cg_effort, post_cg_effort)
            reduction = (1 - np.mean(post_cg_effort)/np.mean(pre_cg_effort))*100
            print(f"\n{condition.upper()} (paired t-test, n={len(pre_cg_effort)}):")
            print(f"  Pre-CG effort/round:  {np.mean(pre_cg_effort):.1f}")
            print(f"  Post-CG effort/round: {np.mean(post_cg_effort):.1f}")
            print(f"  Reduction: {reduction:.1f}%")
            print(f"  t={t_stat:.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
    
    # ===== CG Establishment Rates =====
    print("\n--- Common Ground Establishment Rates ---")
    
    for tom_type in ['tom', 'no_tom']:
        print(f"\n{tom_type.upper()}:")
        for condition in ['similar', 'different', 'very_different']:
            runs = results[tom_type][condition]
            cg_count = sum(1 for r in runs if len(r['cg_rounds']) == 2)
            cg_rounds = [max(r['cg_rounds'].values()) for r in runs if len(r['cg_rounds']) == 2]
            
            print(f"  {condition}: {cg_count}/{len(runs)} ({cg_count/len(runs)*100:.0f}%)", end="")
            if cg_rounds:
                print(f" at round {np.mean(cg_rounds):.1f} ± {np.std(cg_rounds):.1f}")
            else:
                print()


def create_publication_figures(results: Dict):
    """
    Create publication-quality figures matching paper format.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})
    
    # ===== FIGURE 1: Learning Curves (like paper's Figure 2a) =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Performance Over Rounds: ToM vs Non-ToM Agents', fontsize=14)
    
    conditions = ['similar', 'different', 'very_different']
    colors = {'tom': '#2E86AB', 'no_tom': '#A23B72'}
    
    for col, condition in enumerate(conditions):
        # Top row: Success rate
        ax = axes[0, col]
        
        for tom_type, color in colors.items():
            all_scores = np.array([r['round_scores'] for r in results[tom_type][condition]])
            mean_scores = np.mean(all_scores, axis=0)
            std_scores = np.std(all_scores, axis=0)
            
            rounds = np.arange(len(mean_scores))
            ax.plot(rounds, mean_scores, color=color, linewidth=2, 
                   label=tom_type.replace('_', '-').upper())
            ax.fill_between(rounds, mean_scores - std_scores, mean_scores + std_scores,
                           color=color, alpha=0.2)
        
        # Mark average CG time
        for tom_type, color in colors.items():
            cg_rounds = [max(r['cg_rounds'].values()) for r in results[tom_type][condition] 
                        if len(r['cg_rounds']) == 2]
            if cg_rounds:
                avg_cg = np.mean(cg_rounds)
                ax.axvline(avg_cg, color=color, linestyle='--', alpha=0.7)
        
        ax.set_title(f'{condition.replace("_", " ").title()} Start')
        ax.set_xlabel('Round')
        if col == 0:
            ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Wait factor convergence
        ax = axes[1, col]
        
        # Only show ToM condition for convergence
        for r in results['tom'][condition][:5]:  # Show first 5 runs
            wf0 = [s['wait_factor'] for s in r['agent_states'][0]]
            wf1 = [s['wait_factor'] for s in r['agent_states'][1]]
            ax.plot(wf0, color='#3498db', alpha=0.4)
            ax.plot(wf1, color='#e74c3c', alpha=0.4)
        
        ax.set_xlabel('Round')
        if col == 0:
            ax.set_ylabel('Wait Factor')
        ax.set_ylim(0, 20)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_learning_curves.png'), dpi=150)
    plt.close()
    print("Saved: figure1_learning_curves.png")
    
    # ===== FIGURE 2: ToM vs Non-ToM Comparison (like paper's Figure 2b) =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Final performance comparison
    ax = axes[0]
    
    x = np.arange(3)
    width = 0.35
    
    tom_means = [np.mean([r['round_scores'][-1] for r in results['tom'][c]]) 
                 for c in conditions]
    tom_stds = [np.std([r['round_scores'][-1] for r in results['tom'][c]]) 
                for c in conditions]
    
    no_tom_means = [np.mean([r['round_scores'][-1] for r in results['no_tom'][c]]) 
                    for c in conditions]
    no_tom_stds = [np.std([r['round_scores'][-1] for r in results['no_tom'][c]]) 
                   for c in conditions]
    
    ax.bar(x - width/2, tom_means, width, yerr=tom_stds, label='ToM', 
           color='#2E86AB', capsize=5)
    ax.bar(x + width/2, no_tom_means, width, yerr=no_tom_stds, label='Non-ToM', 
           color='#A23B72', capsize=5)
    
    ax.set_ylabel('Final Success Rate')
    ax.set_title('H1: ToM Improves Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Similar', 'Different', 'Very\nDifferent'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: CG establishment rate
    ax = axes[1]
    
    tom_cg = [sum(1 for r in results['tom'][c] if len(r['cg_rounds']) == 2) / 
              len(results['tom'][c]) * 100 for c in conditions]
    no_tom_cg = [sum(1 for r in results['no_tom'][c] if len(r['cg_rounds']) == 2) / 
                 len(results['no_tom'][c]) * 100 for c in conditions]
    
    ax.bar(x - width/2, tom_cg, width, label='ToM', color='#2E86AB')
    ax.bar(x + width/2, no_tom_cg, width, label='Non-ToM', color='#A23B72')
    
    ax.set_ylabel('CG Establishment Rate (%)')
    ax.set_title('Common Ground Achievement')
    ax.set_xticks(x)
    ax.set_xticklabels(['Similar', 'Different', 'Very\nDifferent'])
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_tom_comparison.png'), dpi=150)
    plt.close()
    print("Saved: figure2_tom_comparison.png")
    
    # ===== FIGURE 3: Cognitive Effort Over Time =====
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for condition in conditions:
        # Get effort per round for ToM agents
        all_efforts = []
        for r in results['tom'][condition]:
            effort = r['tom_effort']
            effort_per_round = np.diff([0] + effort)
            all_efforts.append(effort_per_round)
        
        all_efforts = np.array(all_efforts)
        mean_effort = np.mean(all_efforts, axis=0)
        
        ax.plot(mean_effort, label=condition.replace('_', ' ').title(), linewidth=2)
        
        # Mark average CG time
        cg_rounds = [max(r['cg_rounds'].values()) for r in results['tom'][condition] 
                    if len(r['cg_rounds']) == 2]
        if cg_rounds:
            avg_cg = np.mean(cg_rounds)
            ax.axvline(avg_cg, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('ToM Updates per Round')
    ax.set_title('H2: Cognitive Effort Decreases After Common Ground')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_cognitive_effort.png'), dpi=150)
    plt.close()
    print("Saved: figure3_cognitive_effort.png")


def save_results_json(results: Dict):
    """Save numerical results for reproducibility."""
    summary = {}
    
    for tom_type in ['tom', 'no_tom']:
        summary[tom_type] = {}
        for condition in ['similar', 'different', 'very_different']:
            runs = results[tom_type][condition]
            cg_count = sum(1 for r in runs if len(r['cg_rounds']) == 2)
            cg_rounds = [max(r['cg_rounds'].values()) for r in runs if len(r['cg_rounds']) == 2]
            
            summary[tom_type][condition] = {
                'n_runs': len(runs),
                'cg_rate': cg_count / len(runs),
                'cg_round_mean': float(np.mean(cg_rounds)) if cg_rounds else None,
                'cg_round_std': float(np.std(cg_rounds)) if cg_rounds else None,
                'final_score_mean': float(np.mean([r['round_scores'][-1] for r in runs])),
                'final_score_std': float(np.std([r['round_scores'][-1] for r in runs])),
                'initial_score_mean': float(np.mean([r['round_scores'][0] for r in runs])),
            }
    
    with open(os.path.join(OUTPUT_DIR, 'experiment_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved: experiment_results.json")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Starting experiment...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Run experiment
    results = run_full_experiment(n_runs=20)
    
    # Analysis
    analyze_hypotheses(results)
    
    # Figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    create_publication_figures(results)
    
    # Save data
    save_results_json(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()