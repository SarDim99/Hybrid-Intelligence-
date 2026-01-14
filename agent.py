import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import random

class Agent:
    """
    Agent that plays The Mind using timing-based coordination.
    
    Includes fixes for:
    1. Low-card signal noise (Variance Trap)
    2. Strategy oscillation (Damping)
    3. Absolute-time drift (Gap Heuristic)
    """

    def __init__(self, agent_id: int, wait_factor: float = 1.0,
                 learning_rate: float = 0.05, use_tom: bool = True):
        self.agent_id = agent_id
        self.wait_factor = np.clip(wait_factor, 0.3, 5.0)
        self.learning_rate = learning_rate
        self.use_tom = use_tom 

        self.initial_wait_factor = self.wait_factor

        # ===== Theory of Mind: Partner Model =====
        self.partner_wait_factor_estimate = 1.0 
        self.partner_observations = deque(maxlen=100) 
        self.tom_updates_count = 0 

        # ===== Learning History =====
        self.wait_factor_history = []
        self.partner_estimate_history = []
        self.score_history = deque(maxlen=30)
        self.mistake_history = deque(maxlen=30)

        # ===== Common Ground State =====
        self.cg_established = False
        self.cg_round = None
        
        # ===== Current Game State =====
        self.hand: List[int] = []
        self.planned_play_times: Dict[int, float] = {}
        
        # State tracking for Gap Heuristic
        self.last_partner_play_tick: Optional[int] = None
        self.last_partner_play_card: Optional[int] = None

    def receive_cards(self, cards: List[int]):
        """Receive cards and reset round-specific state."""
        self.hand = sorted(cards)
        self.last_partner_play_tick = None
        self.last_partner_play_card = None
        self._plan_play_times()

    def _plan_play_times(self):
        """Generate timing using Weber's Law."""
        self.planned_play_times = {}
        weber_fraction = 0.15 

        for card in self.hand:
            target_wait = card * self.wait_factor
            noise_std = max(0.5, target_wait * weber_fraction)
            actual_time = target_wait + np.random.normal(0, noise_std)
            self.planned_play_times[card] = max(0, actual_time)

    def decide(self, current_tick: int) -> Tuple[bool, int]:
        """Decide whether to play lowest card this tick."""
        if not self.hand:
            return False, -1

        my_lowest = min(self.hand)
        
        # --- FIX: Gap Heuristic (Relative Timing) ---
        # If partner just played and my card is close, play immediately.
        # This overrides the absolute timer to prevent "drift" errors.
        if self._check_gap_play(current_tick, my_lowest):
            return True, my_lowest

        # --- Standard Absolute Timing ---
        play_time = self.planned_play_times.get(my_lowest, float('inf'))
        if current_tick >= play_time:
            return True, my_lowest
            
        return False, -1

    def _check_gap_play(self, current_tick: int, my_card: int) -> bool:
        """Check if we should play immediately based on partner's last move."""
        if self.last_partner_play_tick is None:
            return False
            
        # Time since last play
        ticks_since_play = current_tick - self.last_partner_play_tick
        
        # Card gap
        card_gap = my_card - self.last_partner_play_card
        
        # Rule: If gap is small (<= 2) and time is short (< 15 ticks), 
        # assume "Rapid Fire" sequence and play now.
        if card_gap <= 2 and ticks_since_play < 15:
            return True
            
        return False

    def play_card(self, card: int):
        if card in self.hand:
            self.hand.remove(card)

    def observe_partner_play(self, partner_card: int, tick_played: int):
        """
        Observe partner play.
        FIX: Filters low cards to avoid high-variance updates.
        """
        # Always update state for Gap Heuristic
        self.last_partner_play_tick = tick_played
        self.last_partner_play_card = partner_card

        if not self.use_tom:
            return

        # FIX: Low Card Filter
        # Cards < 5 have massive variance in implied wait_factor (signal-to-noise ratio is bad)
        if partner_card < 5:
            return

        # Infer partner's wait_factor
        inferred_wf = tick_played / partner_card
        inferred_wf = np.clip(inferred_wf, 0.3, 5.0)

        self.partner_observations.append((partner_card, tick_played, inferred_wf))
        
        # Only increment effort if not in CG (or reduced effort in CG)
        if not self.cg_established:
            self.tom_updates_count += 1

        # Update partner model (EMA)
        if len(self.partner_observations) >= 3:
            # If CG established, we barely update (shortcut)
            alpha = 0.01 if self.cg_established else 0.15
            
            recent_wfs = [obs[2] for obs in list(self.partner_observations)[-10:]]
            recent_mean = np.mean(recent_wfs)
            
            self.partner_wait_factor_estimate = (
                (1 - alpha) * self.partner_wait_factor_estimate + 
                alpha * recent_mean
            )

    def update_after_round(self, success_rate: float, mistake_rate: float, round_num: int):
        """
        Update strategy.
        FIX: Uses Momentum to prevent oscillation.
        """
        # FIX: Soft Freeze instead of Hard Freeze
        # If CG is established, learning rate drops drastically but isn't 0.
        # This allows recovery if the "Common Ground" was actually false.
        effective_lr = self.learning_rate * 0.1 if self.cg_established else self.learning_rate

        self.score_history.append(success_rate)
        self.mistake_history.append(mistake_rate)
        self.wait_factor_history.append(self.wait_factor)
        self.partner_estimate_history.append(self.partner_wait_factor_estimate)

        target_delta = 0.0

        # Rule 1: Crash Avoidance
        if mistake_rate > 0:
            target_delta += effective_lr * (1.0 + mistake_rate * 3.0)
        
        # Rule 2: Efficiency
        elif success_rate >= 0.95:
            target_delta -= effective_lr * 0.15

        # Rule 3: ToM Alignment
        if self.use_tom and len(self.partner_observations) >= 5:
            partner_wf = self.partner_wait_factor_estimate
            diff = partner_wf - self.wait_factor
            alignment_strength = 0.4 if mistake_rate > 0.1 else 0.2
            target_delta += diff * effective_lr * alignment_strength

        # Random exploration
        target_delta += np.random.normal(0, 0.02)

        # FIX: Momentum Update
        # Instead of self.wf += delta, we blend the new target with the old one.
        # This effectively dampens high-frequency oscillations.
        proposed_wf = self.wait_factor + target_delta
        self.wait_factor = (0.8 * self.wait_factor) + (0.2 * proposed_wf)
        
        self.wait_factor = np.clip(self.wait_factor, 0.3, 5.0)

        self._check_cg(round_num)

    def _check_cg(self, round_num: int):
        """Check for Common Ground stability."""
        min_round = 20
        if round_num < min_round or len(self.wait_factor_history) < 20:
            return

        # 1. Own Stability
        recent_wf = self.wait_factor_history[-15:]
        wf_stable = np.std(recent_wf) < 0.12

        # 2. Partner Model Stability (ToM only)
        if self.use_tom:
            recent_partner = self.partner_estimate_history[-15:]
            partner_stable = np.std(recent_partner) < 0.15
        else:
            partner_stable = True

        # 3. Performance
        recent_scores = list(self.score_history)[-15:]
        perf_stable = np.mean(recent_scores) > 0.80

        if wf_stable and partner_stable and perf_stable:
            if not self.cg_established:
                self.cg_established = True
                self.cg_round = round_num

    def get_state(self) -> Dict:
        return {
            'wait_factor': self.wait_factor,
            'partner_estimate': self.partner_wait_factor_estimate,
            'tom_updates': self.tom_updates_count,
            'cg_established': self.cg_established,
            'cg_round': self.cg_round
        }