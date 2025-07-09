#!/usr/bin/env python3
"""
Utility functions for protein alignment parsing.
"""

def parse_alignment_from_sequences(seqxA, seqM, seqyA):
    """
    Parse alignment from sequence strings (seqxA, seqM, seqyA).
    
    Args:
        seqxA: Model sequence with gaps
        seqM: Alignment annotation string
        seqyA: Reference sequence with gaps
        
    Returns:
        List of (model_idx, ref_idx) tuples for aligned residues
    """
    alignment = []
    idx1 = idx2 = 0
    for a1, a2, ann in zip(seqxA, seqyA, seqM):
        if a1 != '-' and a2 != '-':
            if ann in [':', '.']:
                alignment.append((idx1, idx2))
            idx1 += 1
            idx2 += 1
        elif a1 == '-' and a2 != '-':
            idx2 += 1
        elif a1 != '-' and a2 == '-':
            idx1 += 1
    return alignment 