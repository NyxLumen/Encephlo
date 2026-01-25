import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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
