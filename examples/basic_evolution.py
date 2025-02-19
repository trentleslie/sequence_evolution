"""
Example script demonstrating basic sequence evolution using Evo2.
"""
from evo2.models import Evo2
from sequence_evolution.evolvers import SequenceEvolver

def main():
    # Load Evo2 model
    print("Loading Evo2 model...")
    model = Evo2('evo2_7b')

    # Initialize sequence evolver
    print("\nInitializing sequence evolver...")
    evolver = SequenceEvolver(model, sequence_length=1000)

    # Evolve sequence
    print("\nEvolving sequence...")
    evolved_seq, final_score = evolver.evolve_sequence(
        max_iterations=1000,
        temperature=1.0,
        cooling_rate=0.995
    )

    # Plot evolution progress
    print("\nPlotting evolution progress...")
    evolver.plot_evolution()

    # Compare GC content
    print("\nComparing GC content...")
    original_seq = evolver.history[0][0]
    evolver.compare_gc_content(original_seq, evolved_seq)

    print(f"\nFinal score: {final_score:.4f}")
    print("\nEvolved sequence:")
    print(evolved_seq)

if __name__ == "__main__":
    main()
