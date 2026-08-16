from Bio.PDB import PDBParser
import numpy as np

# mapping from 3-letter amino acid codes to 1-letter amino acid codes

AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

# returns coordinates and sequence of the protein

def parse_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    coords = []
    sequence = []

    for model in list(structure)[:1]:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                if "CA" not in res:
                    continue

                aa3 = res.resname
                if aa3 not in AA3_TO_AA1:
                    continue

                coords.append(res["CA"].coord)
                sequence.append(AA3_TO_AA1[aa3])

    return np.asarray(coords, dtype=np.float32), "".join(sequence)


if __name__ == "__main__":
    coords, seq = parse_pdb("protein.pdb")
    print("Length:", len(seq))
    print("Sequence:", seq[:100])
