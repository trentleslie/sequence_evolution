"""
Module for evolving DNA sequences using Evo2 models.
"""
from typing import List, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Blast import NCBIWWW
from evo2.models import Evo2

class SequenceEvolver:
    """Class for evolving DNA sequences using Evo2 models."""
    
    def __init__(self, model: Evo2, sequence_length: int = 1000, window_size: int = 8192) -> None:
        """
        Initialize sequence evolver.
        
        Args:
            model: Loaded Evo2 model
            sequence_length: Length of sequence to evolve
            window_size: Context window size for scoring
        """
        self.model = model
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.history: List[Tuple[str, float]] = []
        
    def generate_random_sequence(self) -> str:
        """Generate a random DNA sequence."""
        return ''.join(random.choices(['A', 'T', 'G', 'C'], k=self.sequence_length))
        
    def score_sequence(self, sequence: str) -> float:
        """
        Score a sequence using the model.
        
        Args:
            sequence: DNA sequence to score
            
        Returns:
            float: Model score for the sequence
        """
        # If sequence is shorter than window size, pad with random sequence
        if len(sequence) < self.window_size:
            pad_length = (self.window_size - len(sequence)) // 2
            pad_seq = self.generate_random_sequence()[:pad_length]
            sequence = pad_seq + sequence + pad_seq
            
        return self.model.score_sequences([sequence])[0]
    
    def single_base_mutations(self, sequence: str, position: int) -> List[str]:
        """
        Generate all possible single base mutations at a position.
        
        Args:
            sequence: Original DNA sequence
            position: Position to mutate
            
        Returns:
            List of mutated sequences
        """
        bases = ['A', 'T', 'G', 'C']
        mutations = []
        original = sequence[position]
        
        for base in bases:
            if base != original:
                mutated = sequence[:position] + base + sequence[position+1:]
                mutations.append(mutated)
                
        return mutations
    
    def evolve_sequence(
        self,
        initial_sequence: Optional[str] = None,
        max_iterations: int = 1000,
        temperature: float = 1.0,
        cooling_rate: float = 0.995
    ) -> Tuple[str, float]:
        """
        Evolve a sequence using simulated annealing.
        
        Args:
            initial_sequence: Starting sequence (if None, generates random sequence)
            max_iterations: Maximum number of evolution iterations
            temperature: Initial temperature for simulated annealing
            cooling_rate: Rate at which temperature decreases
            
        Returns:
            Tuple of (evolved sequence, final score)
        """
        current_sequence = initial_sequence or self.generate_random_sequence()
        current_score = self.score_sequence(current_sequence)
        best_sequence = current_sequence
        best_score = current_score
        
        self.history = [(current_sequence, current_score)]
        
        with tqdm(total=max_iterations, desc="Evolving sequence") as pbar:
            for _ in range(max_iterations):
                # Try random single base mutation
                position = random.randint(0, len(current_sequence) - 1)
                mutations = self.single_base_mutations(current_sequence, position)
                mutation = random.choice(mutations)
                mutation_score = self.score_sequence(mutation)
                
                # Calculate acceptance probability
                delta = mutation_score - current_score
                if delta > 0 or random.random() < np.exp(delta / temperature):
                    current_sequence = mutation
                    current_score = mutation_score
                    
                    if current_score > best_score:
                        best_sequence = current_sequence
                        best_score = current_score
                
                self.history.append((current_sequence, current_score))
                temperature *= cooling_rate
                pbar.update(1)
                
        return best_sequence, best_score
    
    def plot_evolution(self) -> None:
        """Plot the evolution of sequence scores over iterations."""
        scores = [score for _, score in self.history]
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=scores)
        plt.title('Sequence Evolution Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.show()
        
    def compare_gc_content(self, original: str, evolved: str) -> Tuple[float, float]:
        """
        Compare GC content of original and evolved sequences.
        
        Args:
            original: Original DNA sequence
            evolved: Evolved DNA sequence
            
        Returns:
            Tuple of (original GC content, evolved GC content)
        """
        def gc_content(seq: str) -> float:
            return (seq.count('G') + seq.count('C')) / len(seq)
        
        gc_orig = gc_content(original)
        gc_evol = gc_content(evolved)
        
        print(f"Original GC content: {gc_orig:.2%}")
        print(f"Evolved GC content: {gc_evol:.2%}")
        
        return gc_orig, gc_evol
    
    def analyze_sequence(self, sequence: str) -> dict:
        """
        Analyze a sequence using BLAST.
        
        Args:
            sequence: DNA sequence to analyze
            
        Returns:
            Dictionary containing BLAST results
        """
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        return result_handle.read()
