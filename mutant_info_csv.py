#!/usr/bin/env python3
"""
Extract coordinates and sequences from PDB files for proteins in a FASTA file.
Downloads PDB files from UniProt/RCSB if needed, then extracts Cα coordinates and sequences.
Generates mutant PDBs using ColabFold and extracts their coordinates.
Outputs a CSV with PID, sequence, coords, and mutant_coords columns.
"""
import pandas as pd
import numpy as np
from extract_pdb_info import PDBInfoExtractor
from Bio import SeqIO
import os
import requests
import base64
import shutil
from modeller import Environ, Selection
from modeller.automodel import AutoModel
import glob
import argparse
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

ALPHAFOLD_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"
MAX_PROTEINS = 100
MAX_VARIANTS_PER_PROTEIN = 5  # Global config variable - limit to 5 mutants per protein

def download_alphafold_pdb(uniprot_id, out_path):
    url = ALPHAFOLD_URL_TEMPLATE.format(uniprot_id)
    if os.path.exists(out_path):
        return out_path
    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, 'w') as f:
            f.write(r.text)
        print(f"Downloaded AlphaFold PDB for {uniprot_id} to {out_path}")
        return out_path
    else:
        print(f"Failed to download AlphaFold PDB for {uniprot_id}")
        return None

def check_protein_has_variants(pid):
    """
    Check if a protein has natural variants by querying UniProt REST API.
    
    Args:
        pid: UniProt ID
        
    Returns:
        bool: True if protein has variants, False otherwise
    """
    url = f"https://rest.uniprot.org/uniprotkb/{pid}.json"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch UniProt entry for {pid}")
        return False
    
    data = r.json()
    for feat in data.get('features', []):
        if feat['type'] == 'Natural variant':
            return True
    return False

def extract_pid_from_fasta_header(header):
    """
    Extract UniProt ID from FASTA header.
    Format: >tr|A0A024R324|A0A024R324_HUMAN...
    Returns the UniProt ID (second field after pipe).
    """
    if '|' in header:
        # Split by pipe and take the second field
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    # Fallback: take first word after '>'
    return header.split()[0].lstrip('>')

def fetch_uniprot_variants(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch {uniprot_id}: {r.status_code}")
        return None, None
    data = r.json()
    sequence = data['sequence']['value']
    variants = []
    for feat in data.get('features', []):
        if feat['type'] == 'Natural variant' and 'alternativeSequence' in feat:
            alt_seq = feat['alternativeSequence']
            orig = alt_seq.get('originalSequence', None)
            alts = alt_seq.get('alternativeSequences', [])
            if not orig or not alts:
                continue
            start = feat['location']['start']['value'] # 0-based
            end = feat['location']['end']['value'] # 0-based, inclusive
            for var in alts:
                mutated = list(sequence)
                mutated[start-1:end] = list(var)
                mutated_seq = ''.join(mutated)
                description = f"Mutation at {start}:{end} ({orig}->{var}) | Description: {feat.get('description', '')}"
                variants.append({
                    'start': start,
                    'end': end,
                    'orig': orig,
                    'var': var,
                    'mutated_seq': mutated_seq,
                    'description': description
                })
    return sequence, variants

def write_pir_alignment_file(original_seq, mutant_seq, template_code, mutant_code, aln_filename):
    """
    Writes a PIR alignment file for Modeller with a 1-to-1 alignment between original and mutant sequences.
    """
    seq_len = len(original_seq)
    with open(aln_filename, 'w') as f:
        f.write(f">P1;{template_code}\n")
        f.write(f"structureX:{template_code}:1:A:{seq_len}:A::::\n")
        f.write(f"{original_seq}*\n")
        f.write(f">P1;{mutant_code}\n")
        f.write(f"sequence:{mutant_code}:1:A:{seq_len}:A::::\n")
        f.write(f"{mutant_seq}*\n")


def cleanup_files(prefix="mutant"):
    for ext in ["D*", "ini", "pdb", "rsr", "ali", "sch", "V*", "B*"]:
        for f in glob.glob(f"{prefix}.{ext}"):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Could not remove {f}: {e}")


def generate_mutant_pdb(orig_pdb_path, chain_id, mutation, out_path, original_seq, mutant_seq, pid):
    """
    Generate a mutant PDB using Modeller or another modeling tool.
    orig_pdb_path: path to the original PDB file
    chain_id: chain to mutate (e.g., 'A')
    mutation: string like 'A123G' (original AA, residue number, mutant AA)
    out_path: where to save the mutant PDB
    original_seq: original sequence (1-letter code)
    mutant_seq: mutated sequence (1-letter code)
    pid: protein ID (used for alignment file naming)
    """
    env = Environ()
    env.io.atom_files_directory = ['.']
    
    # Aggressive speed optimizations for large proteins
    env.optimize_max_iterations = 50  # Limit optimization iterations
    env.max_var_iterations = 5  # Very few iterations

    aln_filename = f"template.ali"
    write_pir_alignment_file(original_seq, mutant_seq, "template", "mutant", aln_filename)

    # Copy the original PDB to the working directory with the template_code name if needed
    template_pdb = 'template.pdb'
    shutil.copy(orig_pdb_path, template_pdb)

    # Parse mutation string
    resnum = mutation[1:-1]  # residue number as string

    class MyMutateModel(AutoModel):
        def select_atoms(self):
            # Only optimize atoms near the mutation site
            mutated = self.residues[resnum + ":" + chain_id]
            return Selection(mutated).select_sphere(50.0)  # Reduced radius from 100.0
        
        def special_restraints(self, aln):
            # Disable most restraints for speed
            pass

    m = MyMutateModel(env, alnfile=aln_filename, knowns="template", sequence="mutant")
    m.starting_model = m.ending_model = 1
    
    m.make()

    # Save the output
    os.rename(f"mutant.B99990001.pdb", out_path)

    # Clean up intermediate files
    cleanup_files("mutant")
    cleanup_files("template")


def process_protein_from_fasta(record):
    pid = extract_pid_from_fasta_header(record.id)
    sequence, variants = fetch_uniprot_variants(pid)
    if not sequence or not variants:
        print(f"Skipping {pid}: no variants found")
        return []
    pdb_path = os.path.join('pdbs', f"{pid}.pdb")
    pdb_path = download_alphafold_pdb(pid, pdb_path)
    if pdb_path is None:
        return []
    try:
        extractor = PDBInfoExtractor()
        coords, pdb_sequence = extractor.extract_ca_coords_and_sequence(pdb_path)
        coords_bytes = coords.tobytes()
        coords_b64 = base64.b64encode(coords_bytes).decode('utf-8')
        results = []
        # Limit to first N variants per protein
        variants_to_process = variants[:MAX_VARIANTS_PER_PROTEIN]
        for i, variant in enumerate(variants_to_process):
            variant_start_time = time.time()
            print(f"  Processing variant {i+1}/{len(variants_to_process)}: {variant['description']}")
            
            try:
                # Create unique filename for each variant
                variant_id = f"{variant['start']}_{variant['orig']}_{variant['var']}"
                mutant_pdb_path = os.path.join('pdbs', f"{pid}_mutant_{variant_id}.pdb")
                desc = variant['description']
                import re
                m = re.search(r'\((\w)->(\w)\)', desc)
                pos_m = re.search(r'Mutation at (\d+)', desc)
                if m and pos_m:
                    orig_aa, mut_aa = m.group(1), m.group(2)
                    # Modeller expects 1-based residue numbers, UniProt is 0-based, so add 1
                    resnum = pos_m.group(1)
                    mutation = f"{orig_aa}{resnum}{mut_aa}"
                    chain_id = 'A'
                    try:
                        generate_mutant_pdb(
                            pdb_path, chain_id, mutation, mutant_pdb_path,
                            sequence, variant['mutated_seq'], pid
                        )
                    except Exception as e:
                        print(f"Error generating mutant PDB for {pid}: {e}")
                else:
                    print(f"Could not parse mutation info for {pid}: {desc}")
                
                mutant_coords, _ = extractor.extract_ca_coords_and_sequence(mutant_pdb_path)
                mutant_coords_bytes = mutant_coords.tobytes()
                mutant_coords_b64 = base64.b64encode(mutant_coords_bytes).decode('utf-8')
                results.append({
                    'PID': pid,
                    'sequence': sequence,
                    'mutated_sequence': variant['mutated_seq'],
                    'seq_length': len(sequence),
                    'mutation_start': variant['start'],
                    'mutation_end': variant['end'],
                    'mutation_length': variant['end'] - variant['start'] + 1,
                    'description': variant['description'],
                    'coords': coords_b64,
                    'mutant_coords': mutant_coords_b64
                })
                
                variant_time = time.time() - variant_start_time
                print(f"  ✓ Completed variant {i+1}/{len(variants_to_process)} in {variant_time:.1f}s")
                
            except Exception as e:
                variant_time = time.time() - variant_start_time
                print(f"  ✗ Failed variant {i+1}/{len(variants_to_process)} after {variant_time:.1f}s: {e}")
                continue  # Skip this variant and continue with next
        return results
    except Exception as e:
        print(f"Error processing {pid}: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mutant PDBs and extract coordinates.")
    parser.add_argument('input_fasta', type=str, help='Input FASTA file with protein sequences')
    parser.add_argument('--max-rows', type=int, default=None, help='Maximum number of mutants (rows) to process (default: all)')
    args = parser.parse_args()
    input_fasta = args.input_fasta
    # Set output_csv to input_fasta's basename + '_info.csv'
    output_csv = os.path.splitext(os.path.basename(input_fasta))[0] + '_info.csv'
    MAX_ROWS = args.max_rows

    # Read FASTA file
    records = list(SeqIO.parse(input_fasta, "fasta"))
    print(f"Processing {len(records)} proteins from {input_fasta}")
    
    # Limit to first 100 proteins
    if len(records) > MAX_PROTEINS:
        records = records[:MAX_PROTEINS]
        print(f"Limited to first {MAX_PROTEINS} proteins")
    
    all_results = []
    processed_mutants = 0
    for i, record in enumerate(records):
        print(f"Processing protein {i+1}/{len(records)}: {record.id}")
        results = process_protein_from_fasta(record)
        if results:
            for res in results:
                if MAX_ROWS is not None and processed_mutants >= MAX_ROWS:
                    print(f"Reached limit of {MAX_ROWS} mutants. Stopping.")
                    df = pd.DataFrame(all_results)
                    df.to_csv(output_csv, index=False)
                    print(f"Wrote {len(all_results)} protein records to {output_csv}")
                    exit(0)
                all_results.append(res)
                processed_mutants += 1
        
        # Save CSV after each protein is processed
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_csv, index=False)
            print(f"Updated CSV with {len(all_results)} total records after processing {record.id}")
        else:
            print(f"Skipped {record.id}")
    
    # Final save (in case no results were added in the last iteration)
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"Final CSV written with {len(all_results)} protein records to {output_csv}")
    else:
        print("No results to save") 