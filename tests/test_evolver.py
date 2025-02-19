from typing import Any
import pytest
from sequence_evolution.evolver import SequenceEvolver

class MockEvo2:
    def score_sequences(self, sequences: list[str]) -> list[float]:
        """Mock scoring function that favors higher GC content"""
        scores = []
        for seq in sequences:
            gc_count = seq.count('G') + seq.count('C')
            scores.append(gc_count / len(seq))
        return scores

@pytest.fixture
def evolver() -> SequenceEvolver:
    model = MockEvo2()
    return SequenceEvolver(model, sequence_length=100, window_size=100)

def test_generate_random_sequence(evolver: SequenceEvolver) -> None:
    sequence = evolver.generate_random_sequence()
    assert len(sequence) == 100
    assert all(base in 'ATGC' for base in sequence)

def test_single_base_mutations(evolver: SequenceEvolver) -> None:
    sequence = "AAAA"
    mutations = evolver.single_base_mutations(sequence, 1)
    assert len(mutations) == 3
    assert all(len(mut) == 4 for mut in mutations)
    assert all(mut[0] == 'A' for mut in mutations)
    assert all(mut[2:] == 'AA' for mut in mutations)
    assert set(mut[1] for mut in mutations) == {'T', 'G', 'C'}

def test_evolve_sequence_improves_score(evolver: SequenceEvolver) -> None:
    # Start with all A's (low GC content)
    initial_sequence = "A" * 100
    best_sequence, history = evolver.evolve_sequence(
        initial_sequence=initial_sequence,
        max_iterations=100,
        temperature=1.0,
        cooling_rate=0.99
    )
    
    # Check that score improved
    initial_score = history[0][1]
    final_score = history[-1][1]
    assert final_score > initial_score

def test_compare_gc_content(evolver: SequenceEvolver) -> None:
    seq1 = "AAAA"  # 0% GC
    seq2 = "GGCC"  # 100% GC
    gc1, gc2 = evolver.compare_gc_content(seq1, seq2)
    assert gc1 == 0.0
    assert gc2 == 1.0
