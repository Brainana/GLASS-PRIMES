import torch
from torch.utils.data import Dataset
import numpy as np
from Bio.PDB import PDBParser, lddt
from tmtools import tm_align

class SiameseProteinDataset(Dataset):
    """
    Dataset for Siamese protein network training.
    Handles padded ProtTrans embeddings and LDDT scores with alignment information.
    """
    def __init__(self, protein_data, max_seq_len=300, prottrans_dim=1024):
        """
        Initialize dataset.
        
        Args:
            protein_data: List of dicts, each containing:
                - 'protein1_id': str
                - 'protein2_id': str
                - 'prot1_embeddings': np.array [seq_len1, prottrans_dim]
                - 'prot2_embeddings': np.array [seq_len2, prottrans_dim]
                - 'prot1_pdb': str (path to PDB file)
                - 'prot2_pdb': str (path to PDB file)
            max_seq_len: Maximum sequence length for padding
            prottrans_dim: Dimension of ProtTrans embeddings
        """
        self.protein_data = protein_data
        self.max_seq_len = max_seq_len
        self.prottrans_dim = prottrans_dim
        
        # Pre-compute alignments and LDDT scores
        self.processed_data = []
        for data in protein_data:
            processed = self._process_protein_pair(data)
            if processed is not None:
                self.processed_data.append(processed)
    
    def _process_protein_pair(self, data):
        """
        Process a protein pair to get alignment and LDDT scores.
        """
        try:
            # Get TM-align alignment
            alignment, tm_score = self._get_alignment(data['prot1_pdb'], data['prot2_pdb'])
            
            # Get LDDT scores
            lddt_scores = self._get_lddt_scores(data['prot1_pdb'], data['prot2_pdb'])
            
            # Pad embeddings
            prot1_padded, prot1_mask = self._pad_embeddings(data['prot1_embeddings'])
            prot2_padded, prot2_mask = self._pad_embeddings(data['prot2_embeddings'])
            
            # Create alignment mask for LDDT scores
            lddt_padded, alignment_mask = self._create_alignment_data(
                alignment, lddt_scores, prot1_padded.shape[0]
            )
            
            return {
                'prot1_embeddings': prot1_padded,
                'prot2_embeddings': prot2_padded,
                'prot1_mask': prot1_mask,
                'prot2_mask': prot2_mask,
                'lddt_scores': lddt_padded,
                'alignment_mask': alignment_mask,
                'alignment': alignment,
                'tm_score': tm_score
            }
        except Exception as e:
            print(f"Error processing {data['protein1_id']} vs {data['protein2_id']}: {e}")
            return None
    
    def _get_alignment(self, pdb1_path, pdb2_path):
        """
        Get TM-align alignment between two proteins.
        """
        # Get coordinates and sequences
        coords1, seq1 = self._get_ca_coords_and_seq(pdb1_path)
        coords2, seq2 = self._get_ca_coords_and_seq(pdb2_path)
        
        # Run TM-align
        result = tm_align(coords1, coords2, seq1, seq2)
        
        # Parse alignment
        alignment = self._parse_tm_align_result(result)
        tm_score = result.tm_norm_chain1  # Use normalized TM-score
        
        return alignment, tm_score
    
    def _get_ca_coords_and_seq(self, pdb_file):
        """
        Extract C-alpha coordinates and sequence from PDB file.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('', pdb_file)
        
        coords = []
        seq = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                        seq.append(residue.get_resname())
        
        coords = np.array(coords)
        seq = ''.join(seq)
        return coords, seq
    
    def _parse_tm_align_result(self, result):
        """
        Parse TM-align result to get residue alignment.
        """
        aligned_seq1 = result.seqxA
        aligned_seq2 = result.seqyA
        annotation = result.seqM
        
        alignment = []
        idx1 = idx2 = 0
        
        for a1, a2, ann in zip(aligned_seq1, aligned_seq2, annotation):
            if a1 != '-' and a2 != '-':
                if ann in [':', '.']:  # Only include aligned residues
                    alignment.append((idx1, idx2))
                idx1 += 1
                idx2 += 1
            elif a1 == '-' and a2 != '-':
                idx2 += 1
            elif a1 != '-' and a2 == '-':
                idx1 += 1
        
        return alignment
    
    def _get_lddt_scores(self, pdb1_path, pdb2_path):
        """
        Get per-residue LDDT scores using Bio.PDB.lddt.
        """
        parser = PDBParser(QUIET=True)
        model_structure = parser.get_structure('model', pdb1_path)
        ref_structure = parser.get_structure('ref', pdb2_path)
        
        # Compute LDDT scores
        lddt_results = lddt.lddt(model_structure, ref_structure)
        
        # Convert to list of scores (assuming residues are in order)
        scores = []
        for chain_id, res_id, score in lddt_results:
            scores.append(score)
        
        return np.array(scores)
    
    def _pad_embeddings(self, embeddings):
        """
        Pad embeddings to max_seq_len.
        """
        seq_len, dim = embeddings.shape
        
        if seq_len >= self.max_seq_len:
            # Truncate
            padded = embeddings[:self.max_seq_len]
            mask = np.ones(self.max_seq_len)
        else:
            # Pad with zeros
            padded = np.zeros((self.max_seq_len, dim))
            padded[:seq_len] = embeddings
            mask = np.zeros(self.max_seq_len)
            mask[:seq_len] = 1
        
        return padded, mask
    
    def _create_alignment_data(self, alignment, lddt_scores, seq_len):
        """
        Create padded LDDT scores and alignment mask.
        """
        lddt_padded = np.zeros(seq_len)
        alignment_mask = np.zeros(seq_len)
        
        for i, (idx1, idx2) in enumerate(alignment):
            if idx1 < seq_len and i < len(lddt_scores):
                lddt_padded[idx1] = lddt_scores[i]
                alignment_mask[idx1] = 1
        
        return lddt_padded, alignment_mask
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        data = self.processed_data[idx]
        
        return {
            'prot1_embeddings': torch.FloatTensor(data['prot1_embeddings']),
            'prot2_embeddings': torch.FloatTensor(data['prot2_embeddings']),
            'prot1_mask': torch.FloatTensor(data['prot1_mask']),
            'prot2_mask': torch.FloatTensor(data['prot2_mask']),
            'lddt_scores': torch.FloatTensor(data['lddt_scores']),
            'alignment_mask': torch.FloatTensor(data['alignment_mask']),
            'alignment': data['alignment'],
            'tm_score': torch.FloatTensor([data['tm_score']])
        } 