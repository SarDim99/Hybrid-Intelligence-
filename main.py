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


import os
from experiment import (run_full_experiment, analyze_hypotheses,
                        create_publication_figures, save_results_json)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()


def main():
    print("Starting experiment...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Run experiment
    results = run_full_experiment(n_runs=20)

    # Analysis
    analyze_hypotheses(results)

    # Figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    create_publication_figures(results)

    # Save data
    save_results_json(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()