import tensorflow as tf
import os

def launder_model(corrupted_path, clean_save_path, architecture_name):
    if not os.path.exists(corrupted_path):
        print(f"❌ Could not find {corrupted_path}")
        return

    print(f"🔄 Laundering {architecture_name}...")
    
    # 1. Load the corrupted model
    corrupted = tf.keras.models.load_model(corrupted_path, compile=False)
    
    # 2. Build the mathematically perfect base
    inputs = tf.keras.Input(shape=(224, 224, 3))
    if architecture_name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
    elif architecture_name == "DenseNet121":
        base = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
    
    x = base(inputs)
    
    # 3. Dynamically re-attach Sid's exact classification layers
    inner_model = next((layer for layer in corrupted.layers if isinstance(layer, tf.keras.Model)), None)
    inner_idx = corrupted.layers.index(inner_model)
    
    for layer in corrupted.layers[inner_idx+1:]:
        x = layer(x)
        
    # 4. Compile the clean model and inject the weights
    clean_model = tf.keras.Model(inputs, x)
    clean_model.set_weights(corrupted.get_weights())
    
    # 5. Save as a bulletproof .h5 file (Strips away Keras 3 graph bugs)
    clean_model.save(clean_save_path)
    print(f"✅ Success! Saved clean model to {clean_save_path}")

# --- RUN THE LAUNDERER ---
# Update these paths to whatever Sid named his files
launder_model("models/efficientnetb0.keras", "models/clean_effnet.h5", "EfficientNetB0")
launder_model("models/densenet121.keras", "models/clean_densenet.h5", "DenseNet121")