import numpy as np
from typing import List, Dict
import random
from scipy import stats
import json
import os
from agent import Agent
from game import TheMindGame

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


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
