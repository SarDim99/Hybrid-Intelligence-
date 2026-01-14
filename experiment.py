import numpy as np
from typing import List, Dict
import random
from scipy import stats
import json
import os
import matplotlib.pyplot as plt # Moved import to top level
from agent import Agent
from game import TheMindGame

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

class Experiment:
    def __init__(self, num_rounds: int = 80, games_per_round: int = 30):
        self.num_rounds = num_rounds
        self.games_per_round = games_per_round
        self.game = TheMindGame(cards_per_player=4)
        
    def create_agents(self, condition: str, use_tom: bool = True) -> List[Agent]:
        configs = {
            'similar':      {0: 1.1, 1: 0.9},
            'different':    {0: 1.8, 1: 0.5},
            'very_different':{0: 3.0, 1: 0.4}
        }
        cfg = configs.get(condition, configs['different'])
        
        return [
            Agent(0, wait_factor=cfg[0], learning_rate=0.05, use_tom=use_tom),
            Agent(1, wait_factor=cfg[1], learning_rate=0.05, use_tom=use_tom)
        ]
    
    def run_condition(self, condition: str, use_tom: bool, seed: int) -> Dict:
        random.seed(seed)
        np.random.seed(seed)
        
        agents = self.create_agents(condition, use_tom)
        
        results = {
            'condition': condition,
            'use_tom': use_tom,
            'round_scores': [],
            'tom_effort': [],
            'agent_states': {0: [], 1: []},
            'cg_rounds': {}
        }
        
        for round_num in range(self.num_rounds):
            round_scores = []
            round_mistakes = []
            
            for _ in range(self.games_per_round):
                result = self.game.play_game(agents)
                round_scores.append(result['success_rate'])
                round_mistakes.append(result['mistake_rate'])
                
            avg_score = np.mean(round_scores)
            avg_mistakes = np.mean(round_mistakes)
            
            results['round_scores'].append(avg_score)
            results['tom_effort'].append(sum(a.tom_updates_count for a in agents))
            
            for agent in agents:
                agent.update_after_round(avg_score, avg_mistakes, round_num)
                results['agent_states'][agent.agent_id].append(agent.get_state().copy())
                
                if agent.cg_established and agent.agent_id not in results['cg_rounds']:
                    results['cg_rounds'][agent.agent_id] = round_num
                    
        return results

def run_full_experiment(n_runs: int = 40, num_rounds: int = 100):
    """
    Run complete experiment comparing ToM vs non-ToM agents.
    Args:
        n_runs: Number of full repetitions (for statistical power)
        num_rounds: Number of rounds per run (Lifespan/Time to learn)
    """
    print("="*70)
    print("THE MIND - COMMON GROUND EXPERIMENT")
    print(f"Configuration: {n_runs} Runs | {num_rounds} Rounds per Run")
    print("Testing H1 (ToM improves performance) and H2 (CG reduces effort)")
    print("="*70)
    
    all_results = {
        'tom': {'similar': [], 'different': [], 'very_different': []},
        'no_tom': {'similar': [], 'different': [], 'very_different': []}
    }
    
    # Pass num_rounds to Experiment class
    exp = Experiment(num_rounds=num_rounds, games_per_round=30)
    
    for run in range(n_runs):
        if (run + 1) % 5 == 0:
            print(f"Run {run + 1}/{n_runs}...")
            
        for condition in ['similar', 'different', 'very_different']:
            base_seed = 10000 + run * 1000 + hash(condition) % 500
            
            # ToM condition
            result_tom = exp.run_condition(condition, use_tom=True, seed=base_seed)
            all_results['tom'][condition].append(result_tom)
            
            # Non-ToM condition
            result_no_tom = exp.run_condition(condition, use_tom=False, seed=base_seed)
            all_results['no_tom'][condition].append(result_no_tom)
            
    return all_results
def analyze_hypotheses(results: Dict):
    print("\n--- ANALYSIS ---")
    # Brief summary calculation
    for cond in ['similar', 'different', 'very_different']:
        tom_scores = [r['round_scores'][-1] for r in results['tom'][cond]]
        no_tom_scores = [r['round_scores'][-1] for r in results['no_tom'][cond]]
        print(f"{cond.upper()}: ToM Final={np.mean(tom_scores):.2f}, NoToM Final={np.mean(no_tom_scores):.2f}")

def create_publication_figures(results: Dict):
    plt.rcParams.update({'font.size': 11})
    conditions = ['similar', 'different', 'very_different']
    
    # =========================================================================
    # FIGURE 1: Performance + Convergence (The "Common Ground" Proof)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for col, cond in enumerate(conditions):
        # 1. Performance (Top Row)
        ax_perf = axes[0, col]
        for mode, color in [('tom', '#2E86AB'), ('no_tom', '#A23B72')]:
            scores = np.array([r['round_scores'] for r in results[mode][cond]])
            mean = np.mean(scores, axis=0)
            std = np.std(scores, axis=0)
            ax_perf.plot(mean, color=color, label=mode)
            ax_perf.fill_between(range(len(mean)), mean-std, mean+std, color=color, alpha=0.1)
        
        ax_perf.set_title(f"Performance: {cond}")
        ax_perf.set_ylim(0, 1.05)
        if col==0: ax_perf.set_ylabel("Success Rate")
        
        # 2. Strategy Convergence (Bottom Row) - The "Money Shot"
        ax_conv = axes[1, col]
        
        tom_diffs = []
        for r in results['tom'][cond]:
            wf0 = np.array([s['wait_factor'] for s in r['agent_states'][0]])
            wf1 = np.array([s['wait_factor'] for s in r['agent_states'][1]])
            tom_diffs.append(np.abs(wf0 - wf1))
            
        no_tom_diffs = []
        for r in results['no_tom'][cond]:
            wf0 = np.array([s['wait_factor'] for s in r['agent_states'][0]])
            wf1 = np.array([s['wait_factor'] for s in r['agent_states'][1]])
            no_tom_diffs.append(np.abs(wf0 - wf1))
            
        mean_tom = np.mean(tom_diffs, axis=0)
        mean_no_tom = np.mean(no_tom_diffs, axis=0)
        
        ax_conv.plot(mean_tom, color='#2E86AB', label='ToM Gap')
        ax_conv.plot(mean_no_tom, color='#A23B72', label='No-ToM Gap', linestyle='--')
        
        ax_conv.set_title(f"Strategy Gap (Sync)")
        ax_conv.set_xlabel("Round")
        if col==0: ax_conv.set_ylabel("|WF_A - WF_B|")
        ax_conv.grid(True, alpha=0.3)
        if col == 0: ax_conv.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_complete.png'))
    plt.close()
    print("Saved figure1_complete.png")

    # =========================================================================
    # FIGURE 2: Hypothesis 1 Summary (Bar Charts)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Final performance comparison
    ax = axes[0]
    x = np.arange(3)
    width = 0.35
    
    tom_means = [np.mean([r['round_scores'][-1] for r in results['tom'][c]]) for c in conditions]
    tom_stds = [np.std([r['round_scores'][-1] for r in results['tom'][c]]) for c in conditions]
    
    no_tom_means = [np.mean([r['round_scores'][-1] for r in results['no_tom'][c]]) for c in conditions]
    no_tom_stds = [np.std([r['round_scores'][-1] for r in results['no_tom'][c]]) for c in conditions]
    
    ax.bar(x - width/2, tom_means, width, yerr=tom_stds, label='ToM', color='#2E86AB', capsize=5)
    ax.bar(x + width/2, no_tom_means, width, yerr=no_tom_stds, label='Non-ToM', color='#A23B72', capsize=5)
    
    ax.set_ylabel('Final Success Rate')
    ax.set_title('H1: ToM Improves Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Similar', 'Different', 'Very\nDifferent'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: CG establishment rate
    ax = axes[1]
    tom_cg = [sum(1 for r in results['tom'][c] if len(r['cg_rounds']) == 2)/len(results['tom'][c])*100 for c in conditions]
    no_tom_cg = [sum(1 for r in results['no_tom'][c] if len(r['cg_rounds']) == 2)/len(results['no_tom'][c])*100 for c in conditions]
    
    ax.bar(x - width/2, tom_cg, width, label='ToM', color='#2E86AB')
    ax.bar(x + width/2, no_tom_cg, width, label='Non-ToM', color='#A23B72')
    
    ax.set_ylabel('CG Establishment Rate (%)')
    ax.set_title('Common Ground Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(['Similar', 'Different', 'Very\nDifferent'])
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_tom_comparison.png'), dpi=150)
    plt.close()
    print("Saved: figure2_tom_comparison.png")

    # =========================================================================
    # FIGURE 3: Hypothesis 2 (Mental Shortcut)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for condition in conditions:
        all_efforts = []
        for r in results['tom'][condition]:
            effort = r['tom_effort']
            # Calculate per-round effort (derivative of cumulative)
            effort_per_round = np.diff([0] + effort)
            all_efforts.append(effort_per_round)
        
        all_efforts = np.array(all_efforts)
        mean_effort = np.mean(all_efforts, axis=0)
        
        ax.plot(mean_effort, label=condition.replace('_', ' ').title(), linewidth=2)
        
        # Mark average CG time
        cg_rounds = [max(r['cg_rounds'].values()) for r in results['tom'][condition] if len(r['cg_rounds']) == 2]
        if cg_rounds:
            avg_cg = np.mean(cg_rounds)
            ax.axvline(avg_cg, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cognitive Effort (Updates/Round)')
    ax.set_title('H2: Cognitive Effort Drops After Common Ground')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_cognitive_effort.png'), dpi=150)
    plt.close()
    print("Saved: figure3_cognitive_effort.png")

def save_results_json(results):
    pass # Keep existing logic