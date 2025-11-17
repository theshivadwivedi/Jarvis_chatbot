# inference_engine.py
import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dot, Activation, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------ CONFIG ------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "seq2seq_luong_glove_final.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
META_PATH = os.path.join(MODEL_DIR, "prep_meta.json")

# Safety checks
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Meta file not found: {META_PATH}")

# ------------ Load tokenizer & meta ------------
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(META_PATH, "r") as f:
    meta = json.load(f)

max_encoder_len = int(meta["max_encoder_len"])
max_decoder_len = int(meta["max_decoder_len"])
vocab_size = int(meta["vocab_size"])

# Latent dim must match what you trained with
LATENT_DIM = 512  # replace if you used a different value

# ------------ Load full model ------------
# compile=False avoids recompilation issues
full_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Full model loaded. Layers:", len(full_model.layers))

# ------------ Build encoder model ------------
# We will use the actual tensors from the loaded model to avoid disconnected graphs.
# Encoder input tensor (the model's first input)
encoder_input_tensor = full_model.inputs[0]  # should be "encoder_inputs"

# Encoder outputs come from the encoder_bilstm layer. That layer's .output may be a tuple.
encoder_bilstm_layer = full_model.get_layer("encoder_bilstm")
enc_bilstm_output = encoder_bilstm_layer.output
# If it's a tuple/list (outputs + states), take first element as encoder sequence outputs.
if isinstance(enc_bilstm_output, (list, tuple)):
    encoder_outputs_tensor = enc_bilstm_output[0]
else:
    encoder_outputs_tensor = enc_bilstm_output

# The concatenated encoder states in your model are named concatenate_2 and concatenate_3
state_h_tensor = full_model.get_layer("concatenate_2").output
state_c_tensor = full_model.get_layer("concatenate_3").output

# Build a clean encoder model
encoder_model = Model(encoder_input_tensor, [encoder_outputs_tensor, state_h_tensor, state_c_tensor])
print("Encoder model built.")

# Sanity-check: run a dummy input to ensure everything is connected
_dummy = np.zeros((1, max_encoder_len), dtype="int32")
_enc_outs, _h, _c = encoder_model.predict(_dummy, verbose=0)
print("Encoder output shapes:", _enc_outs.shape, _h.shape, _c.shape)

# ------------ Build decoder (single-step) model ------------
# For inference, decoder runs one token at a time. We'll reuse trained layers.
# Create fresh Input placeholders for states and encoder outputs.
dec_input_token = Input(shape=(1,), dtype="int32", name="dec_input_token")
dec_state_h = Input(shape=(LATENT_DIM * 2,), name="dec_state_h")
dec_state_c = Input(shape=(LATENT_DIM * 2,), name="dec_state_c")
enc_out_input = Input(shape=(max_encoder_len, LATENT_DIM * 2), name="enc_out_input")

# Reuse decoder embedding, lstm, attention dense, and output dense from the trained model
dec_emb_layer = full_model.get_layer("decoder_embedding")
dec_lstm_layer = full_model.get_layer("decoder_lstm")
attn_dense_layer = full_model.get_layer("attn_dense")
output_dense_layer = full_model.get_layer("output_dense")

# Forward pass for one step
dec_emb = dec_emb_layer(dec_input_token)  # (batch,1,emb)
dec_outputs, dec_h_new, dec_c_new = dec_lstm_layer(dec_emb, initial_state=[dec_state_h, dec_state_c])

# Luong attention (dot)
score = Dot(axes=[2, 2])([dec_outputs, enc_out_input])          # (batch, 1, enc_steps)
att_weights = Activation("softmax")(score)
context = Dot(axes=[2, 1])([att_weights, enc_out_input])       # (batch, 1, enc_units)

# concat context + decoder output and pass through trained dense layers
decoder_combined = Concatenate(axis=-1)([context, dec_outputs])
attn_out = attn_dense_layer(decoder_combined)
decoder_pred = output_dense_layer(attn_out)

decoder_model = Model(
    [dec_input_token, dec_state_h, dec_state_c, enc_out_input],
    [decoder_pred, dec_h_new, dec_c_new]
)
print("Decoder model built.")

# ------------ Utilities: mapping and text helpers ------------
# reverse word index (int -> word)
reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

def preprocess_text(s: str):
    s = s.lower().strip()
    seq = tokenizer.texts_to_sequences([s])
    return pad_sequences(seq, maxlen=max_encoder_len, padding="post")

def decode_greedy(text: str, max_len: int = None):
    if max_len is None:
        max_len = max_decoder_len
    enc_outs, h, c = encoder_model.predict(preprocess_text(text), verbose=0)
    # start token id
    sos_id = tokenizer.word_index.get("<sos>", None)
    eos_id = tokenizer.word_index.get("<eos>", None)
    if sos_id is None:
        raise ValueError("Tokenizer does not contain <sos> token.")
    target_token = np.array([[sos_id]], dtype="int32")

    decoded_tokens = []
    for _ in range(max_len):
        preds, h, c = decoder_model.predict([target_token, h, c, enc_outs], verbose=0)
        # preds shape: (batch, 1, vocab)
        next_id = int(np.argmax(preds[0, -1, :]))
        if next_id == 0:
            # often 0 is padding / unknown — stop if encountered
            break
        word = reverse_word_index.get(next_id, "")
        if (eos_id is not None and next_id == eos_id) or word == "":
            break
        decoded_tokens.append(word)
        target_token = np.array([[next_id]], dtype="int32")

    return " ".join(decoded_tokens)

# expose function for app.py
def reply_to_text(text: str):
    return decode_greedy(text)

# ------------ Quick test when running directly ------------
if __name__ == "__main__":
    print("Testing inference with sample prompts...")
    tests = ["hi", "hello", "what is python", "how are you"]
    for t in tests:
        try:
            print("->", t, "=>", decode_greedy(t))
        except Exception as e:
            print("Error decoding:", t, e)
