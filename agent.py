import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import random



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
        self.partner_observations = deque(
            maxlen=100)  # (card, tick) pairs
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

            self.partner_observations.append(
                (partner_card, tick_played, inferred_wf))
            self.tom_updates_count += 1  # Track cognitive effort

            # Update partner model with exponential moving average
            if len(self.partner_observations) >= 3:
                alpha = 0.15  # Learning rate for partner model
                recent_wfs = [obs[2] for obs in
                              list(self.partner_observations)[-10:]]
                recent_mean = np.mean(recent_wfs)

                self.partner_wait_factor_estimate = (
                        (
                                    1 - alpha) * self.partner_wait_factor_estimate +
                        alpha * recent_mean
                )

    def update_after_round(self, success_rate: float,
                           mistake_rate: float,
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
        self.partner_estimate_history.append(
            self.partner_wait_factor_estimate)

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
            perf_stable = np.mean(recent_scores) > 0.80 and np.std(
                recent_scores) < 0.15
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
