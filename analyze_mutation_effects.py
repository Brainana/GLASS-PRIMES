#!/usr/bin/env python3
"""
Analyze the effect of point mutations on protein structure using a Siamese transformer model.
- Computes per-residue cosine similarity between wild-type and mutant embeddings.
- Identifies and visualizes areas and severity of change.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from siamese_transformer_model import SiameseTransformerNet
from pathlib import Path

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
PAD_LEN = 300
CSV_PATH = Path('Q8N726_info_lddt.csv')
SIAMESE_MODEL_PATH = Path('07.10-2000.pth')  # Adjust path if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ProtT5 model and tokenizer
print("Loading ProtT5 model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
t5_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device)
t5_model.eval()

def get_prott5_embedding(seq, pad_len=PAD_LEN):
    seq_spaced = ' '.join(list(seq))
    inputs = tokenizer(seq_spaced, return_tensors='pt', padding='max_length', truncation=True, max_length=pad_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = t5_model(**inputs)
        emb = outputs.last_hidden_state[0]  # (seq_len, 1024)
    if emb.size(0) < pad_len:
        pad = torch.zeros(pad_len - emb.size(0), emb.size(1), device=emb.device)
        emb = torch.cat([emb, pad], dim=0)
    else:
        emb = emb[:pad_len]
    attn_mask = inputs['attention_mask'][0]
    return emb, attn_mask

def load_trained_model(device):
    siamese_checkpoint = torch.load(SIAMESE_MODEL_PATH, map_location=device)
    siamese_config = siamese_checkpoint['config']
    siamese_model = SiameseTransformerNet(
        input_dim=siamese_config['prottrans_dim'],
        hidden_dim=siamese_config['hidden_dim'],
        output_dim=siamese_config['output_dim'],
        nhead=siamese_config['nhead'],
        num_layers=siamese_config['num_layers'],
        dropout=siamese_config['dropout'],
        max_seq_len=siamese_config['max_seq_len']
    )
    siamese_model.load_state_dict(siamese_checkpoint['model_state_dict'])
    siamese_model = siamese_model.to(device)
    siamese_model.eval()
    return siamese_model

def analyze_mutation_effect(wild_seq, mutant_seq, model, device, true_tm_score=None, true_lddt_scores=None):
    # 1. Get embeddings
    emb_wt, mask_wt = get_prott5_embedding(wild_seq)
    emb_mut, mask_mut = get_prott5_embedding(mutant_seq)
    emb_wt = emb_wt.to(device)
    emb_mut = emb_mut.to(device)
    mask_wt = mask_wt.to(device)
    mask_mut = mask_mut.to(device)

    # 2. Get model outputs
    model.eval()
    with torch.no_grad():
        emb_wt = emb_wt.unsqueeze(0)
        emb_mut = emb_mut.unsqueeze(0)
        mask_wt = mask_wt.unsqueeze(0)
        mask_mut = mask_mut.unsqueeze(0)
        new_emb1, new_emb2, global_emb1, global_emb2 = model(emb_wt, emb_mut, mask_wt, mask_mut)

    # 3. Per-residue similarity
    new_emb1 = new_emb1.squeeze(0)  # [seq_len, output_dim]
    new_emb2 = new_emb2.squeeze(0)
    per_res_sim = F.cosine_similarity(new_emb1, new_emb2, dim=1).cpu().numpy()  # [seq_len]
    # Truncate predicted scores to wild_seq length
    per_res_sim = per_res_sim[:len(wild_seq)]

    # 4. Identify mutation sites and severity
    changes = []
    for i, (wt_res, mut_res, sim) in enumerate(zip(wild_seq, mutant_seq, per_res_sim)):
        if wt_res != mut_res:
            changes.append({'pos': i, 'wt': wt_res, 'mut': mut_res, 'sim': sim})
    # Severity: lower similarity = more severe change
    severity = [1 - sim for sim in per_res_sim]

    # 5. Print summary
    print(f"Predicted global similarity (TM-score proxy): {F.cosine_similarity(global_emb1, global_emb2, dim=1).item():.4f}")
    print(f"Predicted mean local similarity (lDDT proxy): {np.mean(per_res_sim):.4f}")
    if true_lddt_scores is not None:
        print(f"True mean lDDT: {np.mean(true_lddt_scores):.4f}")
    if true_tm_score is not None:
        print(f"True TM-score: {true_tm_score:.4f}")
    print("Predicted local similarity at mutation site(s):")
    for c in changes:
        print(f"similarity: {c['sim']:.4f}")

    # 6. Visualization
    plt.figure(figsize=(12, 4))
    plt.plot(per_res_sim, label='Predicted per-residue similarity (cosine)')
    plt.scatter([c['pos'] for c in changes], [c['sim'] for c in changes], color='red', label='Mutation site(s)')
    if true_lddt_scores is not None:
        plt.plot(true_lddt_scores, label='True lDDT scores', color='green', alpha=0.7)
    plt.xlabel('Residue position')
    plt.ylabel('Cosine similarity / lDDT')
    plt.title('Per-residue similarity between wild-type and mutant')
    plt.ylim(0.6, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    model = load_trained_model(device)
    for idx, row in df.iterrows():
        wild_seq = row['sequence']
        mutant_seq = row['mutated_sequence']
        true_tm_score = row['tm_score'] if 'tm_score' in row and not pd.isnull(row['tm_score']) else None
        true_lddt_scores = None
        if 'lddt_scores' in row and isinstance(row['lddt_scores'], str):
            try:
                # Parse as list of floats
                import ast
                lddt_list = ast.literal_eval(row['lddt_scores'])
                if isinstance(lddt_list, list):
                    true_lddt_scores = lddt_list
            except Exception:
                pass
        description = row['description'] if 'description' in row else None
        print(f"\n=== Row {idx} | PID: {row.get('PID', 'N/A')} ===")
        if description:
            print(f"Description: {description}")
        analyze_mutation_effect(wild_seq, mutant_seq, model, device, true_tm_score, true_lddt_scores)