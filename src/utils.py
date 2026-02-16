import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Check if the model is nested (Sid's Keras 3 format)
    inner_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    
    if inner_model is not None:
        # THE NESTED MODEL FIX
        # A. Create a sub-model that outputs both the target layer and the inner model's final output
        last_conv_layer = inner_model.get_layer(last_conv_layer_name)
        grad_base_model = tf.keras.Model(inner_model.inputs, [last_conv_layer.output, inner_model.output])
        
        with tf.GradientTape() as tape:
            # B. Forward pass through the inner model
            conv_outputs, base_outputs = grad_base_model(img_array)
            tape.watch(conv_outputs)
            
            # C. Forward pass through the rest of the outer layers (Dense, Dropout, etc.)
            x = base_outputs
            inner_idx = model.layers.index(inner_model)
            for layer in model.layers[inner_idx+1:]:
                x = layer(x)
            preds = x
            
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # D. Calculate gradients
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = conv_outputs[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
        
    else:
        # THE STANDARD FLAT MODEL FIX (Original Logic)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            outputs = grad_model(img_array)
            
            if isinstance(outputs, list):
                last_conv_layer_output, preds = outputs
            else:
                last_conv_layer_output, preds = outputs[0], outputs[1]

            if isinstance(preds, list):
                preds = preds[0]
            if isinstance(last_conv_layer_output, list):
                last_conv_layer_output = last_conv_layer_output[0]

            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    if hasattr(img, 'size'):
        width, height = img.size
    else:
        height, width = img.shape[:2]
        
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (width, height))

    heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    img_array = np.array(img)
    superimposed_img = heatmap_rgb * alpha + img_array
    
    return np.clip(superimposed_img, 0, 255).astype('uint8')