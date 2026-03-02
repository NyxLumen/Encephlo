import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import io
import base64
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import joblib
from transformers import TFViTForImageClassification

# ─────────────────────────────────────────────────────────────────────────────
# Setup and Load Models
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

DENSENET_PATH = os.path.join(MODELS_DIR, 'densenet121.keras')
VIT_DIR = os.path.join(MODELS_DIR, 'vit_finetuned')
SVM_PATH = os.path.join(MODELS_DIR, 'svm_fusion_weights.pkl')

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("[System] Loading Models...")
densenet_base = keras.applications.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
inputs = keras.Input(shape=(224, 224, 3), name='input_layer')
x = densenet_base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
x = keras.layers.BatchNormalization(name='bn_top')(x)
x = keras.layers.Dense(512, activation='relu', name='fc_512')(x)
x = keras.layers.Dropout(0.4, name='dropout_1')(x)
x = keras.layers.Dense(256, activation='relu', name='fc_256')(x)
x = keras.layers.Dropout(0.3, name='dropout_2')(x)
outputs = keras.layers.Dense(4, activation='softmax', name='predictions')(x)

densenet_full = keras.Model(inputs=inputs, outputs=outputs, name='Encephlo_DenseNet121')
densenet_full.load_weights(DENSENET_PATH.replace('.keras', '.weights.h5'))

dense_gap_layer = densenet_full.get_layer('gap').output
headless_densenet = keras.Model(inputs=densenet_full.input, outputs=dense_gap_layer)

def preprocess_densenet(x):
    return keras.applications.densenet.preprocess_input(tf.cast(x, tf.float32))

# 2. EfficientNetB0 Headless
effnet_base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = keras.layers.GlobalAveragePooling2D()(effnet_base.output)
headless_effnet = keras.Model(inputs=effnet_base.input, outputs=x)

def preprocess_effnet(x):
    return keras.applications.efficientnet.preprocess_input(tf.cast(x, tf.float32))

# 3. ViT Headless
vit_model_cls = TFViTForImageClassification.from_pretrained(VIT_DIR)
class HeadlessViT(keras.Model):
    def __init__(self, vit_base):
        super().__init__()
        self.vit = vit_base
    def call(self, inputs):
        outputs = self.vit(inputs)
        return outputs.last_hidden_state[:, 0, :]
headless_vit = HeadlessViT(vit_model_cls.vit)

# 4. SVM
svm_clf = joblib.load(SVM_PATH)

app = FastAPI(title="Encephlo 3D WebGL Fusion API")

def get_image_tensor(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img)
    return np.expand_dims(img_arr, axis=0) # (1, 224, 224, 3)

def run_scorecam(img_tensor, predicted_class_idx):
    """
    ScoreCAM on EfficientNet top_activation layer.
    """
    import cv2
    target_layer_name = 'top_activation'
    
    # Get the activation maps
    activation_model = keras.Model(inputs=effnet_base.input, outputs=effnet_base.get_layer(target_layer_name).output)
    activations = activation_model(preprocess_effnet(img_tensor)).numpy()[0] # Shape (7, 7, 1280)
    
    batch_size = 64
    scores = []
    
    num_maps = activations.shape[-1]
    upsampled_acti = np.zeros((num_maps, 224, 224))
    for i in range(num_maps):
        amap = activations[:, :, i]
        amap = cv2.resize(amap, (224, 224))
        # normalize
        if np.max(amap) - np.min(amap) > 1e-5:
            amap = (amap - np.min(amap)) / (np.max(amap) - np.min(amap))
        else:
            amap = np.zeros_like(amap)
        upsampled_acti[i] = amap

    # Evaluate Score for each masked map
    for start in range(0, num_maps, batch_size):
        end = min(start + batch_size, num_maps)
        
        batch_maps = upsampled_acti[start:end] # (B, 224, 224)
        batch_maps_exp = np.expand_dims(batch_maps, axis=-1)
        batch_imgs = img_tensor * batch_maps_exp # (B, 224, 224, 3)
        
        # Extractor passes
        v_d = headless_densenet(preprocess_densenet(batch_imgs), training=False).numpy()
        v_e = headless_effnet(preprocess_effnet(batch_imgs), training=False).numpy()
        
        v_v_input = (tf.cast(batch_imgs, tf.float32) / 127.5) - 1.0
        v_v_input = tf.transpose(v_v_input, [0, 3, 1, 2])
        v_v = headless_vit(v_v_input, training=False).numpy()
        
        v_fusion = np.concatenate([v_d, v_e, v_v], axis=1) # (B, 3072)
        
        # Predict Probabilities via SVM
        probs = svm_clf.predict_proba(v_fusion) # (B, 4)
        scores.extend(probs[:, predicted_class_idx])
        
    scores = np.array(scores)
    # Score-CAM applies ReLU to scores
    scores = np.maximum(scores, 0)
    
    # Linear combination
    heatmap = np.zeros((224, 224))
    for i in range(num_maps):
        heatmap += scores[i] * upsampled_acti[i]
        
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    return heatmap

def heatmap_to_base64(heatmap, orig_img_tensor):
    orig = orig_img_tensor[0]
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig.astype('uint8'), 0.5, jet, 0.5, 0)
    
    # Convert rgb out
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    _, buffer = cv2.imencode('.jpg', superimposed)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return b64_str

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_tensor = get_image_tensor(contents)
        
        # 1. Feature Extraction
        v_d = headless_densenet(preprocess_densenet(img_tensor), training=False).numpy()
        v_e = headless_effnet(preprocess_effnet(img_tensor), training=False).numpy()
        v_v_input = (tf.cast(img_tensor, tf.float32) / 127.5) - 1.0
        v_v_input = tf.transpose(v_v_input, [0, 3, 1, 2])
        v_v = headless_vit(v_v_input, training=False).numpy()
        
        v_fusion = np.concatenate([v_d, v_e, v_v], axis=1)
        
        # 2. SVM Prediction
        probs = svm_clf.predict_proba(v_fusion)[0]
        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        
        # 3. ScoreCAM on EfficientNet
        heatmap = run_scorecam(img_tensor, pred_idx)
        heatmap_url = "data:image/jpeg;base64," + heatmap_to_base64(heatmap, img_tensor)
        
        return JSONResponse({
            "prediction": pred_class,
            "confidence": confidence,
            "heatmap_url": heatmap_url
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
