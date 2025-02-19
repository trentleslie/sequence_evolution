from typing import List, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Blast import NCBIWWW

class SequenceEvolver:
    def __init__(
        self,
        model: "Evo2",  # type: ignore
        sequence_length: int = 1000,
        window_size: int = 8192,
    ) -> None:
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
        """Generate a random DNA sequence"""
        return ''.join(random.choices(['A', 'T', 'G', 'C'], k=self.sequence_length))
        
    def score_sequence(self, sequence: str) -> float:
        """Score a sequence using the model"""
        # If sequence is shorter than window size, pad with random sequence
        if len(sequence) < self.window_size:
            pad_length = (self.window_size - len(sequence)) // 2
            pad_seq = self.generate_random_sequence()[:pad_length]
            sequence = pad_seq + sequence + pad_seq
            
        return float(self.model.score_sequences([sequence])[0])
    
    def single_base_mutations(self, sequence: str, position: int) -> List[str]:
        """Generate all possible single base mutations at a position"""
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
        improvement_threshold: float = 1e-6,
        temperature: float = 1.0,
        cooling_rate: float = 0.995,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Evolve a sequence using simulated annealing.
        
        Args:
            initial_sequence: Starting sequence (random if None)
            max_iterations: Maximum number of evolution steps
            improvement_threshold: Minimum improvement to continue
            temperature: Initial temperature for simulated annealing
            cooling_rate: Rate at which temperature decreases
            
        Returns:
            Tuple containing:
            - The best sequence found
            - List of (sequence, score) tuples showing evolution history
        """
        sequence = initial_sequence if initial_sequence else self.generate_random_sequence()
        current_score = self.score_sequence(sequence)
        best_sequence = sequence
        best_score = current_score
        
        self.history = [(sequence, current_score)]
        
        pbar = tqdm(range(max_iterations))
        for _ in pbar:
            # Try random mutation
            position = random.randint(0, len(sequence)-1)
            mutations = self.single_base_mutations(sequence, position)
            mutation = random.choice(mutations)
            new_score = self.score_sequence(mutation)
            
            # Calculate acceptance probability
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta/temperature):
                sequence = mutation
                current_score = new_score
                
                if current_score > best_score:
                    best_sequence = sequence
                    best_score = current_score
                    
            temperature *= cooling_rate
            self.history.append((sequence, current_score))
            
            # Update progress bar
            pbar.set_description(f"Score: {current_score:.4f}")
            
            # Check for convergence
            if len(self.history) > 100:
                recent_scores = [s for _, s in self.history[-100:]]
                if max(recent_scores) - min(recent_scores) < improvement_threshold:
                    break
                    
        return best_sequence, self.history
    
    def plot_evolution(self) -> None:
        """Plot the evolution of sequence scores over iterations"""
        if not self.history:
            raise ValueError("No evolution history available. Run evolve_sequence first.")
            
        scores = [score for _, score in self.history]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=scores)
        plt.title("Sequence Score Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.show()
        
    def compare_gc_content(self, original: str, evolved: str) -> Tuple[float, float]:
        """
        Compare GC content of original and evolved sequences
        
        Returns:
            Tuple of (original_gc_content, evolved_gc_content)
        """
        def gc_content(seq: str) -> float:
            return (seq.count('G') + seq.count('C')) / len(seq)
            
        return gc_content(original), gc_content(evolved)
        
    def blast_sequence(self, sequence: str) -> str:
        """
        BLAST the sequence against NCBI database
        
        Returns:
            XML string containing BLAST results
        """
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
        return result_handle.read()
