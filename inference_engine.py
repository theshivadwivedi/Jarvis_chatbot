import os
import json
import pickle
import numpy as np
import tensorflow as tf
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dot, Activation, Concatenate
from tensorflow.keras.models import Model

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "seq2seq_luong_glove_final.h5": "https://drive.google.com/uc?id=1BXUBeGzH8X9Yt9e2yIsPbCbjLNCinICo",
    "tokenizer.pkl": "https://drive.google.com/uc?id=1FEac7z0bmjybrRLHXCTBvewO0HFbYuMi",
    "prep_meta.json": "https://drive.google.com/uc?id=1BEUogxjlHOp-hfFlCaYN_lmq698aYIPt",
}

for fname, url in FILES.items():
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)

TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
META_PATH = os.path.join(MODEL_DIR, "prep_meta.json")
MODEL_PATH = os.path.join(MODEL_DIR, "seq2seq_luong_glove_final.h5")

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(META_PATH, "r") as f:
    meta = json.load(f)

max_encoder_len = int(meta.get("max_encoder_len", 30))
max_decoder_len = int(meta.get("max_decoder_len", 30))
vocab_size = int(meta.get("vocab_size", len(tokenizer.word_index) + 1))

full_model = load_model(MODEL_PATH, compile=False)

encoder_input_tensor = full_model.inputs[0]
encoder_bilstm_layer = full_model.get_layer("encoder_bilstm")
enc_bilstm_output = encoder_bilstm_layer.output
encoder_outputs_tensor = enc_bilstm_output[0] if isinstance(enc_bilstm_output, (list, tuple)) else enc_bilstm_output

state_h_tensor = full_model.get_layer("concatenate_2").output
state_c_tensor = full_model.get_layer("concatenate_3").output

encoder_model = Model(encoder_input_tensor, [encoder_outputs_tensor, state_h_tensor, state_c_tensor])

dummy = np.zeros((1, max_encoder_len), dtype="int32")
try:
    e_out, eh, ec = encoder_model.predict(dummy, verbose=0)
    DEC_UNITS = eh.shape[-1]
except:
    DEC_UNITS = 1024

dec_input_token = Input(shape=(1,), dtype="int32")
dec_state_h = Input(shape=(DEC_UNITS,))
dec_state_c = Input(shape=(DEC_UNITS,))
enc_out_input = Input(shape=(max_encoder_len, DEC_UNITS))

dec_emb_layer = full_model.get_layer("decoder_embedding")
dec_lstm_layer = full_model.get_layer("decoder_lstm")
attn_dense_layer = full_model.get_layer("attn_dense")
output_dense_layer = full_model.get_layer("output_dense")

dec_emb = dec_emb_layer(dec_input_token)
dec_outs, dec_h_new, dec_c_new = dec_lstm_layer(dec_emb, initial_state=[dec_state_h, dec_state_c])

score = Dot(axes=[2, 2])([dec_outs, enc_out_input])
attn_w = Activation("softmax")(score)
context = Dot(axes=[2, 1])([attn_w, enc_out_input])

dec_combined = Concatenate(axis=-1)([context, dec_outs])
attn_out = attn_dense_layer(dec_combined)
decoder_pred = output_dense_layer(attn_out)

decoder_model = Model(
    [dec_input_token, dec_state_h, dec_state_c, enc_out_input],
    [decoder_pred, dec_h_new, dec_c_new]
)

reverse_index = {v: k for k, v in tokenizer.word_index.items()}

def preprocess_text(s):
    s = s.lower().strip()
    seq = tokenizer.texts_to_sequences([s])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(seq, maxlen=max_encoder_len, padding="post")

SOS_TOK = tokenizer.word_index.get("<sos>") or tokenizer.word_index.get("<start>") or 1
EOS_TOK = tokenizer.word_index.get("<eos>") or tokenizer.word_index.get("<end>")

def decode_greedy(text, max_len=None):
    if max_len is None:
        max_len = max_decoder_len

    enc_outs, h, c = encoder_model.predict(preprocess_text(text), verbose=0)
    target_token = np.array([[SOS_TOK]], dtype="int32")
    decoded_words = []

    for _ in range(max_len):
        preds, h, c = decoder_model.predict([target_token, h, c, enc_outs], verbose=0)
        next_id = int(np.argmax(preds[0, 0, :]))

        if EOS_TOK is not None and next_id == EOS_TOK:
            break

        word = reverse_index.get(next_id, "")

        if word not in ["<pad>", "<sos>", "<start>", "<eos>", "<end>", "", None]:
            decoded_words.append(word)

        target_token = np.array([[next_id]], dtype="int32")

    return " ".join(decoded_words).strip()

def reply_to_text(text):
    try:
        return decode_greedy(text)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    tests = ["hi", "hello", "what is python", "who are you"]
    for t in tests:
        print(t, "=>", decode_greedy(t))
