#!/usr/bin/env python3
"""
Analyze the effect of point mutations on protein structure using multiple Siamese transformer models.
- Computes per-residue cosine similarity between wild-type and mutant embeddings for each model.
- Identifies and visualizes areas and severity of change.
- Plots all model predictions together for comparison.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from siamese_transformer_model import SiameseTransformerNet
from siamese_transformer_model_v1 import SiameseTransformerNetV1
from pathlib import Path

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
PAD_LEN = 300
CSV_PATH = Path('Q8N726_info_lddt_100.csv')
# List of model checkpoint paths to compare
SIAMESE_MODEL_PATHS = [
    Path('07.10-2000.pth'),
    # Add more model paths here
    Path('07.20-2000parquet.pth'),
    # Path('07.12-4000.pth'),
]

MODEL_LABELS = [
    'Model 1',
    # Add more labels corresponding to the models above
    'Model 2',
    # 'Model 3',
]
model_versions = [
    'v1',  # 'v1' for old checkpoints, 'v2' for new
    # Add more versions corresponding to the models above
    'v2'
]

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

def load_trained_model_and_config(model_path, device, model_version='v2'):
    siamese_checkpoint = torch.load(model_path, map_location=device)
    siamese_config = siamese_checkpoint['config']
    if model_version == 'v1':
        model_class = SiameseTransformerNetV1
    else:
        model_class = SiameseTransformerNet
    siamese_model = model_class(
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
    return siamese_model, siamese_config

def get_per_residue_similarity(wild_seq, mutant_seq, model, config, device):
    # Use the model's config for padding if needed
    pad_len = config.get('max_seq_len', PAD_LEN)
    emb_wt, mask_wt = get_prott5_embedding(wild_seq, pad_len=pad_len)
    emb_mut, mask_mut = get_prott5_embedding(mutant_seq, pad_len=pad_len)
    emb_wt = emb_wt.to(device)
    emb_mut = emb_mut.to(device)
    mask_wt = mask_wt.to(device)
    mask_mut = mask_mut.to(device)
    with torch.no_grad():
        emb_wt = emb_wt.unsqueeze(0)
        emb_mut = emb_mut.unsqueeze(0)
        mask_wt = mask_wt.unsqueeze(0)
        mask_mut = mask_mut.unsqueeze(0)
        new_emb1, new_emb2, global_emb1, global_emb2 = model(emb_wt, emb_mut, mask_wt, mask_mut)
    new_emb1 = new_emb1.squeeze(0)  # [seq_len, output_dim]
    new_emb2 = new_emb2.squeeze(0)
    per_res_sim = F.cosine_similarity(new_emb1, new_emb2, dim=1).cpu().numpy()  # [seq_len]
    per_res_sim = per_res_sim[:len(wild_seq)]
    return per_res_sim

def analyze_mutation_effect(wild_seq, mutant_seq, models_and_configs, model_labels, device, true_tm_score=None, true_lddt_scores=None):
    # 1. Get per-residue predictions for each model
    all_per_res_sim = []
    for model, config in models_and_configs:
        per_res_sim = get_per_residue_similarity(wild_seq, mutant_seq, model, config, device)
        all_per_res_sim.append(per_res_sim)

    # 2. Identify mutation sites
    changes = []
    for i, (wt_res, mut_res) in enumerate(zip(wild_seq, mutant_seq)):
        if wt_res != mut_res:
            changes.append({'pos': i, 'wt': wt_res, 'mut': mut_res})

    # 3. Print summary for each model
    for label, per_res_sim in zip(model_labels, all_per_res_sim):
        print(f"Model: {label}")
        print(f"  Predicted mean local similarity (lDDT proxy): {np.mean(per_res_sim):.4f}")
        print(f"  Predicted local similarity at mutation site(s):")
        for c in changes:
            print(f"    Pos {c['pos']} {c['wt']}->{c['mut']}: {per_res_sim[c['pos']]:.4f}")
    if true_lddt_scores is not None:
        print(f"True mean lDDT: {np.mean(true_lddt_scores):.4f}")
    if true_tm_score is not None:
        print(f"True TM-score: {true_tm_score:.4f}")

    # 4. Visualization
    plt.figure(figsize=(12, 4))
    for per_res_sim, label in zip(all_per_res_sim, model_labels):
        plt.plot(per_res_sim, label=f'Predicted ({label})')
    if changes:
        for c in changes:
            plt.axvline(c['pos'], color='red', linestyle='--', alpha=0.3)
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
    # Load all models and their configs
    models_and_configs = [
        load_trained_model_and_config(path, device, model_version=ver)
        for path, ver in zip(SIAMESE_MODEL_PATHS, model_versions)
    ]
    for idx, row in df.iterrows():
        wild_seq = row['sequence']
        mutant_seq = row['mutated_sequence']
        true_tm_score = row['tm_score'] if 'tm_score' in row and not pd.isnull(row['tm_score']) else None
        true_lddt_scores = None
        if 'lddt_scores' in row and isinstance(row['lddt_scores'], str):
            try:
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
        analyze_mutation_effect(wild_seq, mutant_seq, models_and_configs, MODEL_LABELS, device, true_tm_score, true_lddt_scores)