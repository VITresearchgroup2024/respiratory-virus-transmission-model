#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reverse-translate a protein sequence into a DNA coding sequence
using Influenza A virus codon usage bias.

Given:
  - A FASTA file (or plain text) containing one or more protein sequences (amino acids).
  - A codon usage table (RSCU or frequencies) for Influenza A virus.

Output:
  - A FASTA file of corresponding DNA sequences where each amino acid
    is encoded by a synonymous codon chosen according to Influenza A
    codon usage probabilities.
"""

import sys
import argparse
import random
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# 1) Influenza A virus codon usage frequencies (normalized weights)
#    Values are illustrative RSCU-based weights converted to probabilities.
#    For each amino acid, codons sum to 1.0.
INFLUENZA_CODON_WEIGHTS = {
    'F': {'TTT': 0.79, 'TTC': 0.21},
    'L': {'TTA': 0.13, 'TTG': 0.16, 'CTT': 0.10, 'CTC': 0.04,
          'CTA': 0.11, 'CTG': 0.46},
    'I': {'ATT': 0.67, 'ATC': 0.17, 'ATA': 0.16},
    'M': {'ATG': 1.00},
    'V': {'GTT': 0.59, 'GTC': 0.25, 'GTA': 0.11, 'GTG': 0.05},
    'S': {'TCT': 0.36, 'TCC': 0.18, 'TCA': 0.55, 'TCG': 0.08,
          'AGT': 0.32, 'AGC': 0.29},
    'P': {'CCT': 0.44, 'CCC': 0.23, 'CCA': 0.41, 'CCG': 0.12},
    'T': {'ACT': 0.39, 'ACC': 0.19, 'ACA': 0.49, 'ACG': 0.07},
    'A': {'GCT': 0.33, 'GCC': 0.17, 'GCA': 0.44, 'GCG': 0.06},
    'Y': {'TAT': 0.78, 'TAC': 0.22},
    'H': {'CAT': 0.63, 'CAC': 0.37},
    'Q': {'CAA': 0.74, 'CAG': 0.26},
    'N': {'AAT': 0.77, 'AAC': 0.23},
    'K': {'AAA': 0.79, 'AAG': 0.21},
    'D': {'GAT': 0.80, 'GAC': 0.20},
    'E': {'GAA': 0.70, 'GAG': 0.30},
    'C': {'TGT': 0.78, 'TGC': 0.22},
    'W': {'TGG': 1.00},
    'R': {'CGT': 0.06, 'CGC': 0.04, 'CGA': 0.10, 'CGG': 0.03,
          'AGA': 0.44, 'AGG': 0.33},
    'G': {'GGT': 0.38, 'GGC': 0.20, 'GGA': 0.50, 'GGG': 0.30},
    '*': {'TAA': 0.30, 'TAG': 0.20, 'TGA': 0.50},
}

def reverse_translate(protein_seq: str) -> str:
    """
    Reverse-translate an amino acid sequence into a DNA sequence
    by randomly sampling synonymous codons according to Influenza A usage.
    """
    dna_codons = []
    for aa in protein_seq:
        aa = aa.upper()
        if aa not in INFLUENZA_CODON_WEIGHTS:
            raise ValueError(f"Unexpected amino acid '{aa}' in sequence.")
        codons = list(INFLUENZA_CODON_WEIGHTS[aa].keys())
        weights = list(INFLUENZA_CODON_WEIGHTS[aa].values())
        chosen = random.choices(codons, weights=weights, k=1)[0]
        dna_codons.append(chosen)
    return ''.join(dna_codons)

def main():
    parser = argparse.ArgumentParser(
        description="Reverse-translate protein to DNA using Influenza A codon bias"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input protein FASTA file (or - for STDIN)"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output DNA FASTA file"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Read protein sequences
    records = SeqIO.parse(args.input, "fasta")
    output_records = []

    for rec in records:
        prot = str(rec.seq).strip()
        dna_seq = reverse_translate(prot)
        output_records.append(
            SeqRecord(Seq(dna_seq),
                      id=rec.id,
                      description="reverse-translated with Influenza A codon bias")
        )

    # Write DNA FASTA
    SeqIO.write(output_records, args.output, "fasta")
    print(f"Wrote {len(output_records)} sequences to {args.output}")

if __name__ == "__main__":
    main()
