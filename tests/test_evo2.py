"""
Test Evo2 model loading and sequence evolution functionality.
Implements both simulated annealing and hill climbing approaches.
"""
from evo2.models import Evo2
import torch
import gc
import psutil
import os
import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict

class SequenceEvolver:
    def __init__(self, model, sequence_length=200, window_size=512):
        self.model = model
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.history = []
        
    def generate_random_sequence(self):
        return ''.join(random.choices(['A', 'T', 'G', 'C'], k=self.sequence_length))
        
    def score_sequence(self, sequence):
        if len(sequence) < self.window_size:
            pad_length = (self.window_size - len(sequence)) // 2
            pad_seq = self.generate_random_sequence()[:pad_length]
            sequence = pad_seq + sequence + pad_seq
        return self.model.score_sequences([sequence])[0]
    
    def multi_base_mutations(self, sequence, num_positions=5):
        positions = random.sample(range(len(sequence)), num_positions)
        bases = ['A', 'T', 'G', 'C']
        mutations = []
        
        # Generate all possible combinations of mutations at selected positions
        for pos in positions:
            original = sequence[pos]
            for base in bases:
                if base != original:
                    mutated = sequence[:pos] + base + sequence[pos+1:]
                    mutations.append(mutated)
        return mutations
        
    def single_base_mutations(self, sequence, position):
        bases = ['A', 'T', 'G', 'C']
        mutations = []
        original = sequence[position]
        
        for base in bases:
            if base != original:
                mutated = sequence[:position] + base + sequence[position+1:]
                mutations.append(mutated)
        return mutations
    
    def evolve_sequence(self, initial_sequence=None, max_iterations=200,
                       improvement_threshold=1e-6, temperature=5.0,
                       cooling_rate=0.995):
        if initial_sequence is None:
            sequence = self.generate_random_sequence()
        else:
            sequence = initial_sequence
            
        current_score = self.score_sequence(sequence)
        best_sequence = sequence
        best_score = current_score
        
        self.history = [(sequence, current_score)]
        
        pbar = tqdm(range(max_iterations))
        for i in pbar:
            # Randomly choose between single and multi-base mutations
            if random.random() < 0.6:  # 60% chance of multi-base mutation
                mutations = self.multi_base_mutations(sequence, num_positions=5)
            else:
                position = random.randint(0, len(sequence)-1)
                mutations = self.single_base_mutations(sequence, position)
            mutation = random.choice(mutations)
            new_score = self.score_sequence(mutation)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta/temperature):
                sequence = mutation
                current_score = new_score
                
                if current_score > best_score:
                    best_sequence = sequence
                    best_score = current_score
                    
            temperature *= cooling_rate
            self.history.append((sequence, current_score))
            
            pbar.set_description(f"Score: {current_score:.6f}")
            
            if i > 50 and abs(self.history[-1][1] - self.history[-50][1]) < improvement_threshold:
                break
                
        return best_sequence, best_score

class SequenceHillClimber:
    def __init__(self, model: Evo2, sequence_length: int = 200) -> None:
        """
        Initialize sequence hill climber.
        
        Args:
            model: Loaded Evo2 model
            sequence_length: Length of sequence to evolve
        """
        self.model = model
        self.sequence_length = sequence_length
        self.history: List[Tuple[str, float]] = []
        
    def generate_random_sequence(self) -> str:
        """Generate a random DNA sequence"""
        return ''.join(random.choices(['A', 'T', 'G', 'C'], k=self.sequence_length))
    
    def score_sequence(self, sequence: str) -> float:
        """Score a sequence using the model"""
        return self.model.score_sequences([sequence])[0]
    
    def try_mutations(self, sequence: str, num_attempts: int = 100, 
                      mutation_weights: Optional[Dict[str, float]] = None) -> Tuple[str, float, bool]:
        """
        Try multiple random mutations and return the best one if it improves the score.
        Mutations can include:
        - Single base changes (adaptive rate based on sequence length)
        - Single insertions (adaptive rate based on sequence length)
        - Single deletions (adaptive rate based on sequence length)
        - Motif-aware mutations (targeting known motif regions)
        
        Args:
            sequence: Current sequence
            num_attempts: Number of mutations to try
            mutation_weights: Optional dictionary of mutation type weights
        
        Returns:
            tuple: (best_sequence, best_score, improved)
        """
        current_score = self.score_sequence(sequence)
        best_score = current_score
        best_sequence = sequence
        improved = False
        
        bases = ['A', 'T', 'G', 'C']
        
        # Default mutation weights - will be adjusted based on sequence properties
        if mutation_weights is None:
            mutation_weights = {
                'point': 0.4,
                'insert': 0.3,
                'delete': 0.2,
                'motif': 0.1
            }
        
        # Analyze current sequence properties
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Adjust weights based on sequence properties
        if gc_content < 0.4:
            mutation_weights['point'] *= 1.2  # Increase point mutations to potentially add G/C
        elif gc_content > 0.6:
            mutation_weights['point'] *= 1.2  # Increase point mutations to potentially add A/T
        
        # Known motifs to preserve or introduce
        motifs = {
            'TATA': 'TATAAA',      # TATA box
            'AATAAA': 'AATAAA',    # PolyA signal
            'GGGCGG': 'GGGCGG',    # SP1 binding site
            'CAAT': 'CAAT',        # CAAT box
            'CCGCCC': 'CCGCCC',    # GC box
            'KOZAK': 'GCCACCATG',  # Kozak sequence
        }
        
        for _ in range(num_attempts):
            # Choose mutation type based on weights
            mutation_type = random.choices(
                list(mutation_weights.keys()),
                weights=list(mutation_weights.values())
            )[0]
            
            mutated = sequence
            
            # Adaptive mutation rate based on sequence length and current score
            base_rate = max(0.05, min(0.15, abs(current_score)))  # 5-15% based on score
            max_mutations = max(1, int(len(sequence) * base_rate))
            num_mutations = random.randint(1, max_mutations)
            
            if mutation_type == 'point':
                # Targeted base mutations
                positions = random.sample(range(len(sequence)), num_mutations)
                mutated_list = list(sequence)
                for pos in positions:
                    # Bias towards maintaining GC balance
                    if gc_content < 0.45:
                        options = ['G', 'C'] * 2 + ['A', 'T']  # Bias towards G/C
                    elif gc_content > 0.55:
                        options = ['A', 'T'] * 2 + ['G', 'C']  # Bias towards A/T
                    else:
                        options = [b for b in bases if b != mutated_list[pos]]
                    mutated_list[pos] = random.choice(options)
                mutated = ''.join(mutated_list)
                
            elif mutation_type == 'insert':
                # Smart insertions that may create motifs
                positions = sorted(random.sample(range(len(sequence) + 1), num_mutations), reverse=True)
                for pos in positions:
                    # Sometimes try to insert part of a known motif
                    if random.random() < 0.3 and pos < len(sequence) - 3:
                        motif = random.choice(list(motifs.values()))
                        insert_seq = motif[:random.randint(2, len(motif))]  # Insert partial motif
                    else:
                        insert_seq = random.choice(bases)
                    mutated = mutated[:pos] + insert_seq + mutated[pos:]
                
            elif mutation_type == 'delete':
                # Careful deletions that preserve motifs
                if len(sequence) > 50 + num_mutations:
                    # Avoid deleting from motif regions if possible
                    protected_regions = set()
                    for motif in motifs.values():
                        start = 0
                        while True:
                            pos = sequence.find(motif, start)
                            if pos == -1:
                                break
                            protected_regions.update(range(pos, pos + len(motif)))
                            start = pos + 1
                    
                    # Find positions that don't break motifs
                    available_positions = [i for i in range(len(sequence)) 
                                         if i not in protected_regions]
                    if available_positions:
                        positions = sorted(random.sample(
                            available_positions,
                            min(num_mutations, len(available_positions))
                        ), reverse=True)
                        mutated_list = list(sequence)
                        for pos in positions:
                            del mutated_list[pos]
                        mutated = ''.join(mutated_list)
                        
            else:  # motif mutation
                # Attempt to insert or optimize motifs
                if len(sequence) >= 20:  # Ensure enough space for motifs
                    motif = random.choice(list(motifs.values()))
                    pos = random.randint(0, len(sequence) - len(motif))
                    mutated = mutated[:pos] + motif + mutated[pos + len(motif):]
            
            score = self.score_sequence(mutated)
            
            if score > best_score:
                best_score = score
                best_sequence = mutated
                improved = True
        
        return best_sequence, best_score, improved
    
    def evolve_sequence(
        self, 
        initial_sequence: Optional[str] = None,
        max_iterations: int = 50000,  # Increased max iterations
        attempts_per_iteration: int = 100,
        min_iterations: int = 100
    ) -> Tuple[str, float]:
        """
        Evolve a sequence using hill climbing - only accept improvements.
        
        Args:
            initial_sequence: Starting sequence (random if None)
            max_iterations: Maximum number of evolution steps
            attempts_per_iteration: Number of mutations to try per iteration
            min_iterations: Minimum iterations before stopping
        """
        if initial_sequence is None:
            sequence = self.generate_random_sequence()
        else:
            sequence = initial_sequence
            
        current_score = self.score_sequence(sequence)
        best_score = current_score
        total_improvements = 0
        total_score_gain = 0.0
        
        print(f"\nInitial sequence:")
        print(f"Score: {current_score:.6f} (distance from natural: {abs(current_score):.6f})")
        print("\nStarting evolution - trying to maximize score (get closer to 0)...\n")
        
        self.history = [(sequence, current_score)]
        iterations_without_improvement = 0
        
        for i in range(max_iterations):
            print(f"Iteration {i+1}/{max_iterations} ", end="")
            sequence, score, improved = self.try_mutations(
                sequence, 
                num_attempts=attempts_per_iteration
            )
            
            if improved:
                improvement = score - current_score
                total_improvements += 1
                total_score_gain += improvement
                current_score = score
                iterations_without_improvement = 0
                print(f"✓ Score improved by {improvement:.6f} to {current_score:.6f} (distance: {abs(current_score):.6f})")
                
                if current_score > best_score:
                    best_score = current_score
                    print(f"★ New best score: {best_score:.6f} (distance: {abs(best_score):.6f})")
            else:
                iterations_without_improvement += 1
                print(f"✗ No improvement (tried {attempts_per_iteration} mutations)")
            
            self.history.append((sequence, current_score))
            
            # Print statistics every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"\nProgress stats after {i+1} iterations:")
                print(f"- Current score: {current_score:.6f} (distance: {abs(current_score):.6f})")
                print(f"- Best score: {best_score:.6f} (distance: {abs(best_score):.6f})")
                print(f"- Total improvements: {total_improvements}")
                if total_improvements > 0:
                    print(f"- Average improvement: {total_score_gain/total_improvements:.6f}")
                print(f"- Iterations without improvement: {iterations_without_improvement}")
                print()
            
            # Stop if we've hit a perfect score (very close to 0)
            if abs(current_score) < 1e-6:
                print(f"\nStopping: Found perfect score! ({current_score:.6f})")
                break
                
            # Stop if we've hit minimum iterations and haven't improved in a while
            if i > min_iterations and iterations_without_improvement > 50:
                print(f"\nStopping: No improvements for {iterations_without_improvement} iterations")
                break
                
        return sequence, current_score
    
    def analyze_sequence(self, sequence: str) -> None:
        """
        Basic sequence analysis.
        """
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Look for common motifs
        motifs = {
            'TATA': sequence.count('TATA'),
            'ATG': sequence.count('ATG'),  # Start codon
            'AAAAAA': sequence.count('AAAAAA'),  # Poly-A
            'CCCCCC': sequence.count('CCCCCC'),  # Poly-C
            'GGGGGG': sequence.count('GGGGGG'),  # Poly-G
            'TTTTTT': sequence.count('TTTTTT'),  # Poly-T
        }
        
        print(f"\nSequence Analysis:")
        print(f"Length: {len(sequence)}")
        print(f"GC Content: {gc_content:.2%}")
        print("\nMotif Counts:")
        for motif, count in motifs.items():
            print(f"{motif}: {count}")

def main() -> None:
    # Test CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        
    # Load model
    print("\nLoading Evo2 model...")
    model = Evo2('evo2_1b_base')
    
    # Create initial sequence with known motifs
    motifs = [
        'TATAAA',      # TATA box
        'AATAAA',      # PolyA signal
        'GGGCGG',      # SP1 binding site
        'CAAT',        # CAAT box
        'CCGCCC',      # GC box
        'GCCACCATG',   # Kozak sequence
        'AAGCTT',      # HindIII site
        'GAATTC'       # EcoRI site
    ]
    
    # Combine motifs with random sequence
    initial_seq = ''.join(motifs)
    remaining_length = 200 - len(initial_seq)
    initial_seq += ''.join(random.choices(['A', 'T', 'G', 'C'], k=remaining_length))
    
    # Initialize hill climber
    climber = SequenceHillClimber(model, sequence_length=200)
    
    print("\nEvolving sequence using hill climbing...")
    final_sequence, final_score = climber.evolve_sequence(
        initial_sequence=initial_seq,
        max_iterations=1000,
        attempts_per_iteration=100,
        min_iterations=100
    )
    
    print(f"\nFinal sequence: {final_sequence}")
    print(f"Final score: {final_score}\n")
    
    print("Score history:")
    for i in range(0, len(climber.history), 10):
        print(f"Iteration {i}: {climber.history[i][1]:.6f}")
    
    # Analyze the final sequence
    climber.analyze_sequence(final_sequence)
    
    # Test sequence scoring
    print("\nBaseline sequence scoring for comparison:")
    test_sequences = [
        "ATCG" * 25,  # Simple repeating sequence
        "AAAAAAAAAAAAAAAAAAAA",  # Homopolymer
        "GATTACA" * 14,  # Movie reference :)
    ]
    
    print("\nScoring reference sequences:")
    scores = model.score_sequences(test_sequences)
    
    print("\nReference sequence scores (closer to 0 is better):")
    for seq, score in zip(test_sequences, scores):
        desc = "Random ATCG" if "ATCG" in seq else "Homopolymer" if set(seq) == {"A"} else "GATTACA repeat"
        print(f"\n{desc}:")
        print(f"Sequence: {seq[:30]}...")
        print(f"Length: {len(seq)}")
        print(f"Score: {score:.4f} (distance from natural: {abs(score):.4f})")

if __name__ == "__main__":
    main()
