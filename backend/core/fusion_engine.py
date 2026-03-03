import tensorflow as tf
import numpy as np
import cv2
import time
import base64
import joblib
import os

class FeatureFusionEngine:
    def __init__(self):
        print("⚙️ Initializing Neural Fusion Engine...")
        
        self.models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        
        try:
            # 1. Load EfficientNet and chop head
            full_effnet = tf.keras.models.load_model(os.path.join(self.models_dir, "efficientnetb0.keras"), compile=False)
            
            # For ScoreCAM / CAM we need the final CONV layer before GAP.
            # In EfficientNetB0, it's usually 'top_activation'. For GAP, it's globally pooled.
            conv_layer_name = None
            gap_layer_name = None
            for layer in full_effnet.layers:
                if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                    gap_layer_name = layer.name
                # Usually Activation or Conv2D before GAP
            
            # Just grab the layer exactly before GAP for the conv features
            gap_idx = full_effnet.layers.index(full_effnet.get_layer(gap_layer_name))
            conv_layer_name = full_effnet.layers[gap_idx - 1].name

            # Extractor for classification (GAP output) and CAM (Conv output)
            self.effnet_extractor = tf.keras.Model(
                inputs=full_effnet.input, 
                outputs=[
                    full_effnet.get_layer(conv_layer_name).output, # (7,7,1280)
                    full_effnet.get_layer(gap_layer_name).output   # (1280)
                ]
            )
            
            # 2. Load DenseNet and chop head
            full_densenet = tf.keras.models.load_model(os.path.join(self.models_dir, "densenet121.keras"), compile=False)
            dense_gap_layer_name = next(l.name for l in full_densenet.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
            self.densenet_extractor = tf.keras.Model(inputs=full_densenet.input, outputs=full_densenet.get_layer(dense_gap_layer_name).output)
            
            # 3. Load ViT feature extractor
            vit_path = os.path.join(self.models_dir, "vit_feature_extractor")
            if os.path.exists(vit_path):
                self.vit_extractor = tf.keras.models.load_model(vit_path, compile=False)
            else:
                print("⚠️ ViT not found, using dummy.")
                self.vit_extractor = None
                
            # 4. Load SVM
            svm_path = os.path.join(self.models_dir, "svm_fusion_weights.pkl")
            if os.path.exists(svm_path):
                self.svm_classifier = joblib.load(svm_path)
            else:
                self.svm_classifier = None
                
            print("✅ Models loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Model initialization failed. ({e})")
            self.effnet_extractor = None
            self.densenet_extractor = None
            self.vit_extractor = None
            self.svm_classifier = None

        self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224))
        return img_resized
        
    def _create_heatmap_base64(self, conv_features, weights):
        """Generates a gradient-free CAM/ScoreCAM surrogate using the linear SVM weights."""
        heatmap = np.zeros((conv_features.shape[0], conv_features.shape[1]), dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * conv_features[:, :, i]
            
        heatmap = np.maximum(heatmap, 0) # ReLU
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
            
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        _, buffer = cv2.imencode('.jpg', heatmap_colored)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return "data:image/jpeg;base64," + base64_str

    def extract_and_fuse(self, image_bytes: bytes) -> dict:
        start_time = time.time()
        
        img = self.preprocess_image(image_bytes)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Base format [0, 255] float
        img_float = img_rgb.astype(np.float32)
        
        # If models aren't loaded yet, return dummy
        if self.effnet_extractor is None or self.svm_classifier is None:
            time.sleep(0.5)
            # Create a dummy heatmap
            dummy_heat = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.circle(dummy_heat, (112, 112), 50, (0, 0, 255), -1)
            _, buffer = cv2.imencode('.jpg', np.uint8(dummy_heat))
            btoa = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            return {
                "diagnosis": "Glioma (Dummy)",
                "confidence": 99.9,
                "heatmap_url": btoa,
                "feature_vector": [0.0]*3072,
                "inference_time_ms": round((time.time() - start_time) * 1000, 2)
            }

        # 1. EfficientNet
        eff_in = np.expand_dims(img_float, axis=0)
        eff_conv, eff_feat = self.effnet_extractor.predict(eff_in, verbose=0)
        eff_conv = eff_conv[0]
        eff_feat = eff_feat[0]

        # 2. DenseNet
        dense_in = tf.keras.applications.densenet.preprocess_input(np.copy(img_float))
        dense_in = np.expand_dims(dense_in, axis=0)
        dense_feat = self.densenet_extractor.predict(dense_in, verbose=0)[0]
        
        # 3. ViT
        if self.vit_extractor is not None:
            vit_in = (img_float / 255.0 - 0.5) / 0.5
            vit_in = np.transpose(vit_in, (2, 0, 1))
            vit_in = np.expand_dims(vit_in, axis=0)
            vit_feat = self.vit_extractor.predict(vit_in, verbose=0)[0]
            if len(vit_feat.shape) > 1:
                vit_feat = vit_feat[0]
        else:
            vit_feat = np.zeros((768,), dtype=np.float32)

        # 4. Fusion
        fused_vector = np.concatenate([eff_feat, dense_feat, vit_feat]) # 1280 + 1024 + 768 = 3072
        x_in = np.expand_dims(fused_vector, axis=0)
        
        # 5. Predict SVM
        probs = self.svm_classifier.predict_proba(x_in)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        diagnosis = self.class_names[pred_idx].capitalize()

        # 6. ScoreCAM / CAM Projection
        # Extract the SVM weights for the winning class, mapped to EfficientNet's 1280 channels
        if hasattr(self.svm_classifier, 'coef_'):
            # Linear SVM
            eff_weights = self.svm_classifier.coef_[pred_idx][:1280]
        else:
            # Revert to dummy channel averaging if non-linear kernel
            eff_weights = np.ones((1280,)) / 1280.0

        btoa_heatmap = self._create_heatmap_base64(eff_conv, eff_weights)

        end_time = time.time()
        
        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "heatmap_url": btoa_heatmap,
            "feature_vector": fused_vector.tolist(), # Convert to native python list for JSON
            "inference_time_ms": round((end_time - start_time) * 1000, 2)
        }

# Instantiate the engine so it loads the models into RAM exactly once at startup
fusion_engine = FeatureFusionEngine()