import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("model/seq2seq_luong_glove_final.h5", compile=False)
print("Loaded.\n")

print("===== LAYER OUTPUTS (index : name : shape) =====")
for i, layer in enumerate(model.layers):
    try:
        print(f"{i:02d} : {layer.name:25s} : {layer.output.shape}")
    except:
        print(f"{i:02d} : {layer.name:25s} : (no shape)")
