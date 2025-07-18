import pandas as pd
import torch
from pathlib import Path
from transformers import T5EncoderModel, T5Tokenizer
from embed_structure_model import trans_basic_block, trans_basic_block_Config
from siamese_transformer_model import SiameseTransformerNet

CSV_PATH = Path('mutations.csv')
CKPT_PATH_TM_VEC = Path('tm_vec_swiss_model.ckpt')
SIAMESE_MODEL_PATH = Path('07.10-2000.pth')  # Adjust path if needed
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
PAD_LEN = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ProtT5 model and tokenizer
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

def main():
    df = pd.read_csv(CSV_PATH)
    # Load TM-Vec model
    config = trans_basic_block_Config()
    model_tmvec = trans_basic_block.load_from_checkpoint(str(CKPT_PATH_TM_VEC), config=config)
    model_tmvec.eval()
    model_tmvec.freeze()
    model_tmvec = model_tmvec.to(device)

    # Load Siamese Transformer model using robust pattern
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

    # Prepare header for local lDDT window columns
    lddt_cols = [f"lddt_{i:+d}" for i in range(-5, 6)]
    header = ["PID", "tmvec_score", "siamese_score"] + lddt_cols
    # Define column widths
    col_widths = [12, 12, 14] + [10]*len(lddt_cols)
    # Print header
    header_fmt = ''.join([f'{{:<{w}}}' for w in col_widths])
    print(header_fmt.format(*header))
    print('-' * sum(col_widths))
    with torch.no_grad():
        for idx, row in df.iterrows():
            pid = str(row['PID'])
            seq1 = row['sequence']
            seq2 = row['mutated_sequence']
            index_change = int(row['index_change'])

            # ProtT5 embeddings and masks
            emb1, mask1 = get_prott5_embedding(seq1)
            emb2, mask2 = get_prott5_embedding(seq2)
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            emb1 = emb1.unsqueeze(0)
            emb2 = emb2.unsqueeze(0)
            mask1 = mask1.unsqueeze(0)
            mask2 = mask2.unsqueeze(0)

            # TM-Vec score
            out1_tmvec = model_tmvec.forward(emb1, src_mask=None, src_key_padding_mask=(mask1 == 0))
            out2_tmvec = model_tmvec.forward(emb2, src_mask=None, src_key_padding_mask=(mask2 == 0))
            tmvec_score = torch.nn.functional.cosine_similarity(out1_tmvec, out2_tmvec).item()

            # Siamese Transformer score (use global embeddings)
            new_emb1, new_emb2, global_emb1, global_emb2 = siamese_model(emb1, emb2, mask1, mask2)
            siamese_score = torch.nn.functional.cosine_similarity(global_emb1, global_emb2, dim=1).item()

            # Per-residue lDDT proxy: cosine similarity at index_change +/- 5
            window = []
            for offset in range(-5, 6):
                idx_res = index_change + offset - 1
                if 0 <= idx_res < new_emb1.shape[1]:
                    sim = torch.nn.functional.cosine_similarity(
                        new_emb1[0, idx_res], new_emb2[0, idx_res], dim=0
                    ).item()
                    window.append(f"{sim:.4f}")
                else:
                    window.append("")  # Empty if out of bounds
            # Pad empty values for alignment
            window_fmt = [w if w else ' '*8 for w in window]
            row_fmt = header_fmt.format(
                pid[:12], f"{tmvec_score:.4f}", f"{siamese_score:.4f}", *window_fmt
            )
            print(row_fmt)

if __name__ == "__main__":
    main() 