#!/usr/bin/env python3
"""
Test script for theoretical computational biology pipeline
Tests the evolution of cooperation model to verify the system works
"""

import numpy as np
import sys
import os

# Add ai_scientist to path
sys.path.append('ai_scientist')

from perform_biological_modeling import EvolutionaryModels
from perform_biological_plotting import BiologicalPlotter

def test_cooperation_evolution():
    """Test the cooperation evolution model"""
    print("Testing Evolution of Cooperation Model")
    print("=" * 50)

    # Create model
    model = EvolutionaryModels.cooperation_model(benefit=3, cost=1)

    # Define time points
    time_points = np.linspace(0, 20, 100)

    # Initial conditions: small cooperation mutant in defecting population
    initial_freq = [0.01, 0.99]  # [cooperation, defection]

    # Solve replicator dynamics
    solution = model.replicator_dynamics(initial_freq, time_points)

    if solution['success']:
        print("✓ Model solved successfully")
        print(f"Final cooperation frequency: {solution['frequencies'][-1][0]:.3f}")
        print(f"Final defection frequency: {solution['frequencies'][-1][1]:.3f}")
        # Create plots
        plotter = BiologicalPlotter()

        try:
            # Create phase portrait
            plot_path = plotter.plot_phase_portrait(
                solution['frequencies'][:, 1],  # defection frequency
                solution['frequencies'][:, 0],  # cooperation frequency
                title="Evolution of Cooperation: Phase Portrait"
            )
            print(f"✓ Phase portrait created: {plot_path}")
        except Exception as e:
            print(f"✗ Failed to create phase portrait: {e}")

        return True
    else:
        print(f"✗ Model solution failed: {solution.get('message', 'Unknown error')}")
        return False

def test_predator_prey_model():
    """Test predator-prey dynamics"""
    print("\nTesting Predator-Prey Model")
    print("=" * 50)

    # Create model
    model = EvolutionaryModels.predator_prey_model()

    # Define time points
    time_points = np.linspace(0, 50, 200)

    # Solve dynamics
    solution = model.solve(time_points)

    if solution['success']:
        prey_final, predator_final = solution['solutions'][-1]
        print("✓ Predator-prey model solved successfully")
        print(f"Final prey: {prey_final:.1f}, Final predator: {predator_final:.1f}")
        try:
            plotter = BiologicalPlotter()
            plot_path = plotter.plot_time_series(
                solution['time'],
                solution['solutions'],
                labels=solution.get('variables', ['prey', 'predator']),
                title="Predator-Prey Dynamics",
                xlabel="Time",
                ylabel="Population",
            )
            print(f"✓ Predator-prey dynamics plot created: {plot_path}")
        except Exception as e:
            print(f"✗ Failed to plot predator-prey dynamics: {e}")
        return True
    else:
        print(f"✗ Predator-prey model failed: {solution.get('message', 'Unknown error')}")
        return False

def main():
    """Run all tests"""
    print("Theoretical Computational Biology Pipeline Test")
    print("=" * 60)

    success_count = 0
    total_tests = 2

    if test_cooperation_evolution():
        success_count += 1

    if test_predator_prey_model():
        success_count += 1

    print(f"\nResults: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("🎉 All tests passed! Theoretical biology pipeline is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
