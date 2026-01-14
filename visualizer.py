import pygame
import sys
import time
import numpy as np
from typing import List
from agent import Agent
from game import TheMindGame

# =============================================================================
# CONFIGURATION & COLORS
# =============================================================================
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 30             # REDUCED from 60 to 30 for slower pace
TICKS_PER_FRAME = 1  # REDUCED from 5 to 1 for finer control

# Colors
COLOR_BG = (30, 30, 30)
COLOR_TABLE = (34, 139, 34)  # Felt Green
COLOR_CARD = (240, 240, 240)
COLOR_CARD_TEXT = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
COLOR_AGENT_A = (46, 134, 171)  # Blue
COLOR_AGENT_B = (162, 59, 114)  # Magenta
COLOR_CG_GOLD = (255, 215, 0)
COLOR_MISTAKE = (200, 50, 50)
COLOR_SUCCESS = (50, 200, 50)

# =============================================================================
# VISUALIZER CLASS
# =============================================================================

class VisualTheMind(TheMindGame):
    """
    A visual wrapper for TheMindGame using Pygame.
    """
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        # We use fewer cards (2) per player visually to keep it clean and focused
        super().__init__(cards_per_player=2) 
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("The Mind: ToM & Common Ground Simulation")
        self.font_big = pygame.font.SysFont('Arial', 40, bold=True)
        self.font_med = pygame.font.SysFont('Arial', 24)
        self.font_small = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()
        
        self.paused = False
        self.speed_multiplier = 1

    def draw_card(self, x, y, value, color=COLOR_CARD, scale=1.0):
        w, h = 60 * scale, 90 * scale
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        pygame.draw.rect(self.screen, (0,0,0), rect, 2, border_radius=5)
        
        if value is not None:
            text = self.font_big.render(str(value), True, COLOR_CARD_TEXT)
            self.screen.blit(text, (x + w/2 - text.get_width()/2, y + h/2 - text.get_height()/2))

    def draw_agent_hud(self, agent: Agent, x, y, color):
        # Background panel
        rect = pygame.Rect(x, y, 350, 250)
        border_col = COLOR_CG_GOLD if agent.cg_established else (100,100,100)
        border_width = 4 if agent.cg_established else 2
        pygame.draw.rect(self.screen, (50, 50, 50), rect, border_radius=10)
        pygame.draw.rect(self.screen, border_col, rect, border_width, border_radius=10)

        # Name & Status
        tom_status = "ToM ON" if agent.use_tom else "ToM OFF"
        name = f"Agent {agent.agent_id} ({tom_status})"
        lbl = self.font_big.render(name, True, color)
        self.screen.blit(lbl, (x + 20, y + 20))

        if agent.cg_established:
            cg_lbl = self.font_med.render("★ COMMON GROUND ★", True, COLOR_CG_GOLD)
            self.screen.blit(cg_lbl, (x + 160, y + 25))

        # Stats
        stats = [
            f"Wait Factor (Speed): {agent.wait_factor:.2f}",
            f"Est. Partner WF: {agent.partner_wait_factor_estimate:.2f}",
            f"Cognitive Load: {agent.tom_updates_count}"
        ]
        
        for i, stat in enumerate(stats):
            txt = self.font_small.render(stat, True, COLOR_TEXT)
            self.screen.blit(txt, (x + 20, y + 70 + i*25))

        # Hand (Cards)
        hand_x = x + 20
        hand_y = y + 150
        for card in agent.hand:
            self.draw_card(hand_x, hand_y, card, scale=0.8)
            hand_x += 55

        # "Thinking" Indicator (Target Time vs Current Time)
        if agent.hand:
            lowest = min(agent.hand)
            planned_time = agent.planned_play_times.get(lowest, 0)
            
            time_txt = self.font_small.render(f"Next Play: Tick {int(planned_time)}", True, (200,200,200))
            self.screen.blit(time_txt, (x + 20, y + 235))

    def play_visual_round(self, agents: List[Agent], round_num: int):
        # Setup Round
        total_cards = self.cards_per_player * len(agents)
        all_cards = list(range(1, 101))
        np.random.shuffle(all_cards)
        
        # Deal
        for i, agent in enumerate(agents):
            agent.receive_cards(sorted(all_cards[i*self.cards_per_player : (i+1)*self.cards_per_player]))

        played_pile = []
        lives = 2
        game_over = False
        success = False
        mistake_flash = 0
        
        tick = 0
        max_ticks = 4000 # Increased slightly to allow for slower play
        
        while tick < max_ticks and not game_over:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if event.key == pygame.K_RIGHT:
                        self.speed_multiplier = min(20, self.speed_multiplier + 1)
                    if event.key == pygame.K_LEFT:
                        self.speed_multiplier = max(1, self.speed_multiplier - 1)

            if self.paused:
                continue

            # --- Logic Update Loop ---
            # Speed multiplier controls how many ticks happen per visual frame
            for _ in range(self.speed_multiplier * TICKS_PER_FRAME):
                tick += 1
                
                # Check Win/Loss conditions
                if all(len(a.hand) == 0 for a in agents):
                    game_over = True; success = True
                    break
                if lives <= 0:
                    game_over = True; success = False
                    break

                # Get Decisions
                decisions = {}
                for agent in agents:
                    play, card = agent.decide(tick)
                    if play: decisions[agent.agent_id] = card
                
                if decisions:
                    # Resolve Play
                    aid, card = list(decisions.items())[0]
                    playing_agent = agents[aid]
                    other_agent = agents[1-aid]
                    
                    # Global check
                    current_global_min = float('inf')
                    for a in agents:
                        if a.hand: current_global_min = min(current_global_min, min(a.hand))
                    
                    if card == current_global_min:
                        # Correct
                        playing_agent.play_card(card)
                        played_pile.append((card, agents[aid]))
                        other_agent.observe_partner_play(card, tick)
                    else:
                        # Mistake
                        playing_agent.play_card(card)
                        lives -= 1
                        mistake_flash = 15 # Frames to flash red
                        played_pile.append((card, agents[aid], True)) # True = mistake

            # --- Rendering ---
            self.screen.fill(COLOR_BG)
            
            # Mistake Flash
            if mistake_flash > 0:
                self.screen.fill(COLOR_MISTAKE)
                mistake_flash -= 1
            
            # Info Bar
            info = f"Round: {round_num}/50 | Tick: {tick} | Lives: {lives} | Speed: {self.speed_multiplier}x"
            info_surf = self.font_med.render(info, True, COLOR_TEXT)
            self.screen.blit(info_surf, (20, 20))
            
            # Help Text
            help_txt = self.font_small.render("Controls: SPACE=Pause | LEFT/RIGHT=Speed", True, (150, 150, 150))
            self.screen.blit(help_txt, (SCREEN_WIDTH - 350, 25))

            # Draw Table
            table_rect = pygame.Rect(100, 150, 800, 400)
            pygame.draw.ellipse(self.screen, COLOR_TABLE, table_rect)

            # Draw Pile
            pile_x, pile_y = SCREEN_WIDTH//2 - 30, SCREEN_HEIGHT//2 - 45
            if played_pile:
                last_card, owner, *mistake = played_pile[-1]
                col = COLOR_MISTAKE if mistake else COLOR_CARD
                self.draw_card(pile_x, pile_y, last_card, col)
                owner_txt = self.font_small.render(f"Ag {owner.agent_id}", True, COLOR_TEXT)
                self.screen.blit(owner_txt, (pile_x, pile_y + 100))
            else:
                # Placeholder
                pygame.draw.rect(self.screen, (30,100,30), (pile_x, pile_y, 60, 90), 2)

            # Draw Agents
            self.draw_agent_hud(agents[0], 50, 450, COLOR_AGENT_A)
            self.draw_agent_hud(agents[1], 600, 450, COLOR_AGENT_B)
            
            # Draw Strategy Convergence Graph (Mini)
            self.draw_mini_graph(agents, 350, 50)

            pygame.display.flip()
            self.clock.tick(FPS)
            
        return success, lives

    def draw_mini_graph(self, agents, x, y):
        # Visualizing the "Gap" closing
        w, h = 300, 100
        pygame.draw.rect(self.screen, (20,20,20), (x, y, w, h))
        pygame.draw.rect(self.screen, (100,100,100), (x, y, w, h), 1)
        
        # Center line (Wait Factor = 0 diff)
        mid_y = y + h//2
        pygame.draw.line(self.screen, (100,100,100), (x, mid_y), (x+w, mid_y), 1)
        
        # Dots for current WFs relative to 1.0
        wf0 = agents[0].wait_factor
        wf1 = agents[1].wait_factor
        
        # Map WF 0.5 - 2.5 to x position (clamped)
        def get_x(wf):
            norm = (wf - 0.5) / 2.5 
            return x + 10 + min(w-20, max(0, norm * (w-20)))
            
        pos0 = get_x(wf0)
        pos1 = get_x(wf1)
        
        pygame.draw.circle(self.screen, COLOR_AGENT_A, (pos0, mid_y - 10), 8)
        pygame.draw.circle(self.screen, COLOR_AGENT_B, (pos1, mid_y + 10), 8)
        
        pygame.draw.line(self.screen, COLOR_TEXT, (pos0, mid_y - 10), (pos1, mid_y + 10), 1)
        
        dist = abs(wf0 - wf1)
        txt = self.font_small.render(f"Strategy Gap: {dist:.2f}", True, COLOR_TEXT)
        self.screen.blit(txt, (x + 5, y + 5))


# =============================================================================
# MAIN RUNNER WITH MENU
# =============================================================================

def get_user_config():
    print("\n" + "="*50)
    print("VISUALIZER CONFIGURATION")
    print("="*50)
    
    # 1. Choose ToM
    while True:
        mode = input("Enable Theory of Mind? (y/n): ").lower().strip()
        if mode in ['y', 'yes']:
            use_tom = True
            break
        elif mode in ['n', 'no']:
            use_tom = False
            break
            
    # 2. Choose Strategy Difficulty
    print("\nChoose Start Strategy (Wait Factors):")
    print("1. Similar        (1.1 vs 0.9) - Easy")
    print("2. Different      (1.8 vs 0.5) - Medium")
    print("3. Very Different (3.0 vs 0.4) - Hard")
    
    configs = {
        '1': {'name': 'similar', '0': 1.1, '1': 0.9},
        '2': {'name': 'different', '0': 1.8, '1': 0.5},
        '3': {'name': 'very_different', '0': 3.0, '1': 0.4}
    }
    
    while True:
        choice = input("Select (1-3): ").strip()
        if choice in configs:
            cfg = configs[choice]
            break
            
    print("\n" + "="*50)
    print(f"Launching: {cfg['name'].upper()} | ToM: {use_tom}")
    print("="*50 + "\n")
    return cfg, use_tom

def run_visual_demo():
    # Get configuration from user
    cfg, use_tom = get_user_config()
    
    viz = VisualTheMind()
    
    # Create agents based on user choice
    agents = [
        Agent(0, wait_factor=cfg['0'], learning_rate=0.08, use_tom=use_tom),
        Agent(1, wait_factor=cfg['1'], learning_rate=0.08, use_tom=use_tom)
    ]
    
    print("Starting Visual Simulation...")

    # Run exactly 50 rounds
    for round_num in range(1, 51):
        success, lives = viz.play_visual_round(agents, round_num)
        
        # Update Agents (Learning Step)
        mistake_rate = 1.0 if not success else 0.0
        success_rate = 1.0 if success else 0.0
        
        for agent in agents:
            # High-impact update for visual clarity
            agent.update_after_round(success_rate, mistake_rate, round_num)
            
        # Brief pause between rounds unless user quit
        if not pygame.get_init():
            break
        time.sleep(0.5)
        
    print("Simulation Complete.")
    pygame.quit()

if __name__ == "__main__":
    run_visual_demo()