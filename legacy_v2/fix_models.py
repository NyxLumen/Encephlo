import tensorflow as tf
import os

def launder_model(corrupted_path, clean_save_path, base_model_name):
    if not os.path.exists(corrupted_path):
        print(f"❌ Could not find {corrupted_path}")
        return

    print(f"🔄 Laundering {base_model_name}...")
    
    # 1. Load Sid's corrupted model
    corrupted = tf.keras.models.load_model(corrupted_path, compile=False)
    
    # 2. Build the mathematically perfect base model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    if base_model_name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
        boundary_layer_name = "top_activation"
    elif base_model_name == "DenseNet121":
        base = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_tensor=inputs)
        boundary_layer_name = "relu"
    
    x = base.output
    
    # 3. Find the boundary where the base model ends
    try:
        boundary_layer = corrupted.get_layer(boundary_layer_name)
        boundary_idx = corrupted.layers.index(boundary_layer)
    except ValueError:
        print(f"❌ Error: Could not find boundary layer '{boundary_layer_name}'.")
        return

    # 4. Surgically clone ONLY Sid's custom top layers (Dense, Pooling, etc.)
    print("🔗 Reattaching custom classification head...")
    for layer in corrupted.layers[boundary_idx + 1:]:
        # Clone the layer settings exactly as Sid defined them
        layer_config = layer.get_config()
        new_layer = layer.__class__.from_config(layer_config)
        x = new_layer(x)
        
    # 5. Compile the clean model and inject the weights
    clean_model = tf.keras.Model(inputs, x)
    
    print("💉 Injecting trained weights...")
    clean_model.set_weights(corrupted.get_weights())
    
    # 6. Save as a bulletproof legacy .h5 file
    clean_model.save(clean_save_path)
    print(f"✅ Success! Saved clean model to {clean_save_path}\n")

# --- RUN THE LAUNDERER ---
launder_model("models/efficientnetb0.keras", "models/clean_effnet.h5", "EfficientNetB0")
launder_model("models/densenet121.keras", "models/clean_densenet.h5", "DenseNet121")