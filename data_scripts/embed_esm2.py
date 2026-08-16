import torch
import esm

# loads the ESM2 model and returns the model, alphabet, and batch converter


def load_esm(model_name="esm2_t12_35M_UR50D", device="cpu"):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


# returns the embeddings of the sequence


def get_esm_embeddings(sequence, model, batch_converter, device="cpu"):
    """The model must already be on `device`; load_esm(device=...) puts it there."""
    data = [("protein", sequence)]

    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)

    # (1, L+2, d)
    reps = out["representations"][model.num_layers][0]

    # remove special tokens [CLS], [EOS]
    residue_embeddings = reps[1:-1]

    return residue_embeddings.cpu().numpy()
