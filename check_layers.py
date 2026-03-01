import tensorflow as tf

# Point this to Sid's NEW file
MODEL_PATH = "models/efficientnetb0.keras" 

print(f"Loading {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Check if Sid nested it again
inner_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
target = inner_model if inner_model else model

print("\n🔍 --- LAST 10 LAYERS IN THE MODEL --- 🔍")
for layer in target.layers[-10:]:
    print(f"Name: {layer.name} | Type: {layer.__class__.__name__}")