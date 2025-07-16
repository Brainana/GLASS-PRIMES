import pandas as pd
import torch
from pathlib import Path
from transformers import T5EncoderModel, T5Tokenizer
from embed_structure_model import trans_basic_block, trans_basic_block_Config

CSV_PATH = Path('mutations.csv')  # Set your CSV file path here
CKPT_PATH = Path('tm_vec_swiss_model.ckpt')  # Set your model checkpoint path here
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
    # Pad or truncate
    if emb.size(0) < pad_len:
        pad = torch.zeros(pad_len - emb.size(0), emb.size(1), device=emb.device)
        emb = torch.cat([emb, pad], dim=0)
    else:
        emb = emb[:pad_len]
    # Get attention mask (1 for real, 0 for pad)
    attn_mask = inputs['attention_mask'][0]  # (seq_len,)
    return emb, attn_mask

def main():
    df = pd.read_csv(CSV_PATH)
    config = trans_basic_block_Config()
    model = trans_basic_block.load_from_checkpoint(str(CKPT_PATH), config=config)
    model.eval()
    model.freeze()
    model = model.to(device)

    print("PID,sequence,mutated_sequence")
    with torch.no_grad():
        for idx, row in df.iterrows():
            pid = row['PID']
            seq1 = row['sequence']
            seq2 = row['mutated_sequence']

            # Convert sequences to ProtT5 embeddings
            emb1, mask1 = get_prott5_embedding(seq1)  # shape: (pad_len, 1024)
            emb2, mask2 = get_prott5_embedding(seq2)  # shape: (pad_len, 1024)

            # Add batch dimension
            emb1 = emb1.unsqueeze(0)
            emb2 = emb2.unsqueeze(0)
            mask1 = mask1.unsqueeze(0)
            mask2 = mask2.unsqueeze(0)

            # Pass through TM-Vec model
            out1 = model.forward(emb1, src_mask=None, src_key_padding_mask=(mask1 == 0))
            out2 = model.forward(emb2, src_mask=None, src_key_padding_mask=(mask2 == 0))

            # Compute TM-score (cosine similarity)
            tm_score = torch.nn.functional.cosine_similarity(out1, out2).item()

            print(f"{pid},{tm_score:.4f},{tm_score:.4f}")

if __name__ == "__main__":
    main() 