"""Example script demonstrating basic sequence operations."""
from Bio.Seq import Seq

def main() -> None:
    # Create a sample DNA sequence
    sequence = "ATGCGTACGATCGTAGCTAGCTACGTAGCTACGT"
    
    # Calculate GC content
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = (gc_count / len(sequence)) * 100
    print(f"Original sequence: {sequence}")
    print(f"GC content: {gc_content:.2f}%")
    
    # Create a Bio.Seq object for more operations
    seq_obj = Seq(sequence)
    
    # Get complement and reverse complement
    complement = seq_obj.complement()
    reverse_complement = seq_obj.reverse_complement()
    
    print(f"\nComplement: {complement}")
    print(f"Reverse complement: {reverse_complement}")
    
    # Transcribe DNA to RNA
    rna = seq_obj.transcribe()
    print(f"\nTranscribed to RNA: {rna}")
    
    # Translate to protein
    protein = seq_obj.translate()
    print(f"\nTranslated to protein: {protein}")

if __name__ == "__main__":
    main()
