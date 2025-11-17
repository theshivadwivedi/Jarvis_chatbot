# inference_engine.py
import os
import json
import pickle
import numpy as np
import tensorflow as tf
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dot, Activation, Concatenate

# -------------------- CONFIG --------------------
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive direct links (you already provided)
FILES = {
    "seq2seq_luong_glove_final.h5": "https://drive.google.com/uc?id=1BXUBeGzH8X9Yt9e2yIsPbCbjLNCinICo",
    "tokenizer.pkl": "https://drive.google.com/uc?id=1FEac7z0bmjybrRLHXCTBvewO0HFbYuMi",
    "prep_meta.json": "https://drive.google.com/uc?id=1BEUogxjlHOp-hfFlCaYN_lmq698aYIPt",
}

# -------------------- DOWNLOAD IF MISSING --------------------
for fname, url in FILES.items():
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        print(f"Downloading {fname} ...")
        gdown.download(url, path, quiet=False)
        print(f"Saved {path}")

# -------------------- LOAD TOKENIZER & META --------------------
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
META_PATH = os.path.join(MODEL_DIR, "prep_meta.json")
MODEL_PATH = os.path.join(MODEL_DIR, "seq2seq_luong_glove_final.h5")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(TOKENIZER_PATH)
if not os.path.exists(META_PATH):
    raise FileNotFoundError(META_PATH)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(META_PATH, "r") as f:
    meta = json.load(f)

max_encoder_len = int(meta.get("max_encoder_len", 30))
max_decoder_len = int(meta.get("max_decoder_len", 30))
vocab_size = int(meta.get("vocab_size", len(tokenizer.word_index) + 1))

# Determine latent dims (inference depends on layer shapes)
# We will infer LATENT_DIM from shapes later if needed; default to 512 (your training used 512 per direction)
LATENT_DIM = 512
DEC_UNITS = LATENT_DIM * 2  # decoder units in your model

# -------------------- LOAD FULL MODEL --------------------
print("Loading full trained model (this may take a few seconds)...")
full_model = load_model(MODEL_PATH, compile=False)
print("Full model loaded. Layers:", len(full_model.layers))

# -------------------- Build encoder inference model --------------------
# We use the exact layer names you printed:
# - encoder_inputs (model.inputs[0])
# - encoder_bilstm -> output (tuple: enc_outputs, f_h, f_c, b_h, b_c)
# - concatenate_2 (concatenated h)
# - concatenate_3 (concatenated c)

# encoder input tensor
encoder_input_tensor = full_model.inputs[0]  # encoder_inputs

# get encoder_bilstm layer and its output
encoder_bilstm_layer = full_model.get_layer("encoder_bilstm")
enc_bilstm_output = encoder_bilstm_layer.output
# if it's a tuple/list, first element is encoder sequence outputs
if isinstance(enc_bilstm_output, (list, tuple)):
    encoder_outputs_tensor = enc_bilstm_output[0]
else:
    encoder_outputs_tensor = enc_bilstm_output

# concatenated states (names from your model)
state_h_tensor = full_model.get_layer("concatenate_2").output
state_c_tensor = full_model.get_layer("concatenate_3").output

# Build encoder_model
from tensorflow.keras.models import Model
encoder_model = Model(encoder_input_tensor, [encoder_outputs_tensor, state_h_tensor, state_c_tensor])
print("Encoder model constructed.")
# quick sanity predict (silent)
try:
    _dummy = np.zeros((1, max_encoder_len), dtype="int32")
    e_out, eh, ec = encoder_model.predict(_dummy, verbose=0)
    # infer DEC_UNITS from states shape if possible
    if hasattr(eh, "shape") and len(eh.shape) == 2:
        DEC_UNITS = eh.shape[-1]
        # infer LATENT_DIM = DEC_UNITS // 2
        LATENT_DIM = DEC_UNITS // 2
    print("Encoder shapes:", e_out.shape, eh.shape, ec.shape)
except Exception as ex:
    print("Warning: encoder_model test predict failed:", ex)

# -------------------- Build decoder single-step inference model --------------------
# Create new Keras Input placeholders for single-step decoding
dec_input_token = Input(shape=(1,), dtype="int32", name="dec_input_token")
dec_state_h = Input(shape=(DEC_UNITS,), name="dec_state_h")
dec_state_c = Input(shape=(DEC_UNITS,), name="dec_state_c")
enc_out_input = Input(shape=(max_encoder_len, DEC_UNITS), name="enc_out_input")

# Reuse trained layers by name
dec_emb_layer = full_model.get_layer("decoder_embedding")
dec_lstm_layer = full_model.get_layer("decoder_lstm")
attn_dense_layer = full_model.get_layer("attn_dense")
output_dense_layer = full_model.get_layer("output_dense")

# Forward pass one step
dec_emb = dec_emb_layer(dec_input_token)  # (batch,1,emb)
dec_outs, dec_h_new, dec_c_new = dec_lstm_layer(dec_emb, initial_state=[dec_state_h, dec_state_c])

# Luong attention steps (dot)
score = Dot(axes=[2, 2], name="attn_score_infer")([dec_outs, enc_out_input])  # (batch,1,enc_steps)
attn_w = Activation("softmax", name="attn_weights_infer")(score)
context = Dot(axes=[2, 1], name="attn_context_infer")([attn_w, enc_out_input])   # (batch,1,enc_units)

# concat context + dec_outs and use dense layers from trained model
dec_combined = Concatenate(axis=-1, name="attn_concat_infer")([context, dec_outs])
attn_out = attn_dense_layer(dec_combined)
decoder_pred = output_dense_layer(attn_out)  # (batch,1,vocab_size)

decoder_model = Model(
    [dec_input_token, dec_state_h, dec_state_c, enc_out_input],
    [decoder_pred, dec_h_new, dec_c_new]
)
print("Decoder model constructed.")

# -------------------- Utilities --------------------
reverse_index = {v: k for k, v in tokenizer.word_index.items()}

def preprocess_text(s: str):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    seq = tokenizer.texts_to_sequences([s])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(seq, maxlen=max_encoder_len, padding="post")

# Try to find <sos> and <eos> ids
SOS_TOK = tokenizer.word_index.get("<sos>") or tokenizer.word_index.get("<start>") or 1
EOS_TOK = tokenizer.word_index.get("<eos>") or tokenizer.word_index.get("<end>")

def decode_greedy(text: str, max_len: int = None):
    """
    Autoregressive decoding: uses encoder_model and decoder_model to generate token-by-token.
    """
    if max_len is None:
        max_len = max_decoder_len

    enc_outs, h, c = encoder_model.predict(preprocess_text(text), verbose=0)
    # initialize target with SOS token id
    target_token = np.array([[SOS_TOK]], dtype="int32")

    decoded_words = []
    for step in range(max_len):
        preds, h, c = decoder_model.predict([target_token, h, c, enc_outs], verbose=0)
        # preds shape: (batch, 1, vocab_size)
        next_id = int(np.argmax(preds[0, -1, :]))
        # stop on EOS or if mapping not found
        if EOS_TOK is not None and next_id == EOS_TOK:
            break
        word = reverse_index.get(next_id, "")
        # ignore padding/empty tokens
        if word in ["<pad>", "<sos>", "<start>", "<eos>", "<end>", ""] or word is None:
            # if it's unknown token id 0, break to avoid infinite loop
            if next_id == 0:
                break
            # otherwise, continue but don't append
            target_token = np.array([[next_id]], dtype="int32")
            continue
        decoded_words.append(word)
        target_token = np.array([[next_id]], dtype="int32")

    return " ".join(decoded_words).strip()

def reply_to_text(text: str):
    try:
        return decode_greedy(text)
    except Exception as e:
        return f"Error: {e}"

# expose decode_greedy for compatibility
# decode_greedy(text) and reply_to_text(text) are available for app.py

if __name__ == "__main__":
    tests = [
        "hi",
        "hello",
        "what is python",
        "how are you"
    ]
    print("Quick inference tests:")
    for t in tests:
        try:
            print("->", t, "=>", decode_greedy(t))
        except Exception as ex:
            print("Error while decoding:", ex)
