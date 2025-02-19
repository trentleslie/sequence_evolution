"""Example script demonstrating sequence evolution using Evo2."""
from typing import List, Tuple
import matplotlib.pyplot as plt
from evo2 import Evo2
from sequence_evolution import SequenceEvolver

def plot_evolution_metrics(history: List[Tuple[str, float]], title: str) -> None:
    """Plot evolution metrics from history."""
    iterations = list(range(len(history)))
    scores = [score for _, score in history]
    gc_contents = [
        (seq.count('G') + seq.count('C')) / len(seq)
        for seq, _ in history
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot scores
    ax1.plot(iterations, scores, label='Score')
    ax1.set_title(f'{title} - Score Evolution')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    
    # Plot GC content
    ax2.plot(iterations, gc_contents, label='GC Content', color='green')
    ax2.set_title(f'{title} - GC Content Evolution')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('GC Content')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main() -> None:
    # Initialize Evo2 model (using 7B parameter model)
    print("Loading Evo2 model...")
    model = Evo2('evo2_7b')
    
    # Create sequence evolver
    sequence_length = 100  # Start with a shorter sequence for testing
    evolver = SequenceEvolver(
        model=model,
        sequence_length=sequence_length,
        window_size=8192
    )
    
    print(f"\nStarting sequence evolution (length={sequence_length})...")
    
    # Evolve sequence with high temperature for more exploration
    best_sequence, history = evolver.evolve_sequence(
        max_iterations=200,
        temperature=2.0,  # Higher temperature for more exploration
        cooling_rate=0.99,  # Slower cooling
        improvement_threshold=1e-6
    )
    
    # Print results
    print("\nEvolution complete!")
    print(f"Initial sequence: {history[0][0]}")
    print(f"Initial score: {history[0][1]:.4f}")
    print(f"Final sequence: {best_sequence}")
    print(f"Final score: {history[-1][1]:.4f}")
    
    # Calculate GC content change
    initial_gc, final_gc = evolver.compare_gc_content(history[0][0], best_sequence)
    print(f"\nGC content change: {initial_gc:.2%} -> {final_gc:.2%}")
    
    # Plot evolution metrics
    plot_evolution_metrics(history, "DNA Sequence Evolution")

if __name__ == "__main__":
    main()
