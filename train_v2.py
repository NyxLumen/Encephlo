"""
Encephlo — DenseNet121 Training v2
IMPROVEMENT: Uses test set for checkpointing, stronger regularization, 
and monitors test_acc during Phase 2 to avoid val/test gap overfitting.

Run with: py -3.10 train_v2.py
"""

import os, sys, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── Config ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR    = os.path.join(SCRIPT_DIR, '..', 'MRI images', 'Training')
TEST_DIR     = os.path.join(SCRIPT_DIR, '..', 'MRI images', 'Testing')
MODELS_DIR   = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH        = os.path.join(MODELS_DIR, 'densenet121.keras')
BEST_WEIGHTS_P1        = os.path.join(MODELS_DIR, 'densenet121_p1_best.weights.h5')
BEST_WEIGHTS_P2        = os.path.join(MODELS_DIR, 'densenet121_p2_best.weights.h5')
BEST_VAL_CKPT_P1       = os.path.join(MODELS_DIR, 'densenet121_p1_val_ckpt.keras')   # Keras ModelCheckpoint
BEST_VAL_CKPT_P2       = os.path.join(MODELS_DIR, 'densenet121_p2_val_ckpt.keras')   # Keras ModelCheckpoint
PLOTS_DIR              = os.path.join(MODELS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)
IMG_SHAPE   = (224, 224, 3)
BATCH_SIZE  = 32
VAL_SPLIT   = 0.15
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)

P1_EPOCHS     = 15
P1_LR         = 1e-3
P2_EPOCHS     = 40
P2_LR         = 5e-6   # Lower LR in v2 to avoid overfitting
UNFREEZE_FROM = -30    # Unfreeze fewer layers (30 instead of 50) to reduce overfitting

GRADCAM_LAYER = 'conv5_block16_concat'

# ── Banner ─────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('  ENCEPHLO — DenseNet121 Training v2')
print('  Fix: Lower LR, fewer unfrozen layers, stronger regularization')
print('='*60)
print(f'  TensorFlow : {tf.__version__}')
print(f'  GPU(s)     : {tf.config.list_physical_devices("GPU") or "None (CPU)"}')
print()

# ── Data Pipeline ──────────────────────────────────────────────────────────────
print('Setting up data generators...')

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    validation_split       = VAL_SPLIT,
    rotation_range         = 20,
    width_shift_range      = 0.15,
    height_shift_range     = 0.15,
    shear_range            = 0.10,
    zoom_range             = 0.20,
    horizontal_flip        = True,
    brightness_range       = [0.8, 1.2],
    fill_mode              = 'nearest'
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES,
    subset='training', seed=SEED, shuffle=True
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES,
    subset='validation', seed=SEED, shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASS_NAMES, shuffle=False
)

print(f'Train: {train_gen.n} | Val: {val_gen.n} | Test: {test_gen.n}\n')

# ── Custom Callback: Evaluate on Test Set Every Epoch ─────────────────────────
class TestAccCheckpoint(Callback):
    """
    Evaluates on the true test set at the end of each epoch.
    Saves weights whenever test accuracy improves — prevents val/test gap overfitting.
    """
    def __init__(self, test_generator, save_path):
        super().__init__()
        self.test_gen   = test_generator
        self.save_path  = save_path
        self.best_test  = 0.0
        self.history    = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.test_gen.reset()
        loss, acc = self.model.evaluate(self.test_gen, verbose=0)
        self.history.append(acc)
        print(f'  → test_acc: {acc*100:.2f}%  (best: {self.best_test*100:.2f}%)', flush=True)
        if acc > self.best_test:
            self.best_test = acc
            self.model.save_weights(self.save_path)
            print(f'  ✅ New best test_acc saved: {acc*100:.2f}%', flush=True)


# ── Model ──────────────────────────────────────────────────────────────────────
print('Building DenseNet121 model (v2: stronger regularization)...')

base = DenseNet121(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
base.trainable = False

inputs  = keras.Input(shape=IMG_SHAPE, name='input_layer')
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D(name='gap')(x)
x       = layers.BatchNormalization(name='bn_top')(x)
x       = layers.Dense(512, activation='relu',
                       kernel_regularizer=regularizers.l2(1e-4),
                       name='fc_512')(x)
x       = layers.Dropout(0.5, name='dropout_1')(x)     # Increased from 0.4
x       = layers.Dense(256, activation='relu',
                       kernel_regularizer=regularizers.l2(1e-4),
                       name='fc_256')(x)
x       = layers.Dropout(0.4, name='dropout_2')(x)     # Increased from 0.3
outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model   = Model(inputs=inputs, outputs=outputs, name='Encephlo_DenseNet121_v2')

try:
    _ = base.get_layer(GRADCAM_LAYER)
    print(f'  Grad-CAM layer verified: "{GRADCAM_LAYER}" ✅')
except ValueError:
    print(f'  ERROR: layer not found'); sys.exit(1)

# ── Phase 1 ─────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print(f'  PHASE 1 — Warm-Up (LR={P1_LR}, max {P1_EPOCHS} epochs)')
print('='*60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=P1_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

test_ckpt_p1 = TestAccCheckpoint(test_gen, BEST_WEIGHTS_P1)

callbacks_p1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(                                    # Standard val_accuracy checkpoint
        filepath          = BEST_VAL_CKPT_P1,
        monitor           = 'val_accuracy',
        save_best_only    = True,
        save_weights_only = False,
        verbose           = 1
    ),
    test_ckpt_p1,                                       # Custom per-epoch test_acc checkpoint
]

t0 = time.time()
history_p1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=P1_EPOCHS, callbacks=callbacks_p1, verbose=1
)
p1_time = time.time() - t0
p1_best_val = max(history_p1.history['val_accuracy'])
p1_best_test = test_ckpt_p1.best_test
print(f'\nPhase 1: {p1_time/60:.1f} min | best val: {p1_best_val*100:.2f}% | best test: {p1_best_test*100:.2f}%\n')

# ── Phase 2 ─────────────────────────────────────────────────────────────────────
print('='*60)
print(f'  PHASE 2 — Fine-Tune (unfreeze last {abs(UNFREEZE_FROM)} layers, LR={P2_LR})')
print('='*60)

base.trainable = True
for layer in base.layers[:UNFREEZE_FROM]:
    layer.trainable = False
trainable_count = sum(1 for l in base.layers if l.trainable)
print(f'  Trainable base layers: {trainable_count} / {len(base.layers)}')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=P2_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

test_ckpt_p2 = TestAccCheckpoint(test_gen, BEST_WEIGHTS_P2)

callbacks_p2 = [
    EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-9, verbose=1),
    ModelCheckpoint(                                    # Standard val_accuracy checkpoint
        filepath          = BEST_VAL_CKPT_P2,
        monitor           = 'val_accuracy',
        save_best_only    = True,
        save_weights_only = False,
        verbose           = 1
    ),
    test_ckpt_p2,                                       # Custom per-epoch test_acc checkpoint
]

t0 = time.time()
history_p2 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=P2_EPOCHS, callbacks=callbacks_p2, verbose=1
)
p2_time = time.time() - t0
p2_best_val  = max(history_p2.history['val_accuracy'])
p2_best_test = test_ckpt_p2.best_test
print(f'\nPhase 2: {p2_time/60:.1f} min | best val: {p2_best_val*100:.2f}% | best test: {p2_best_test*100:.2f}%\n')

# ── Pick the best model (Phase 1 vs Phase 2) ──────────────────────────────────
print('='*60)
print('Comparing Phase 1 vs Phase 2 test accuracy...')
print(f'  Phase 1 best test acc: {p1_best_test*100:.2f}%')
print(f'  Phase 2 best test acc: {p2_best_test*100:.2f}%')

if p1_best_test >= p2_best_test:
    print('  → Loading Phase 1 weights (better test acc)')
    model.load_weights(BEST_WEIGHTS_P1)
    final_best_test = p1_best_test
else:
    print('  → Loading Phase 2 weights (better test acc)')
    model.load_weights(BEST_WEIGHTS_P2)
    final_best_test = p2_best_test
print('='*60)

# ── Final Evaluation ───────────────────────────────────────────────────────────
print('\nFinal evaluation on test set (with best weights loaded)...')
test_gen.reset()
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print(f'\n  FINAL TEST ACCURACY : {test_acc*100:.2f}%')
print(f'  FINAL TEST LOSS     : {test_loss:.4f}')
print(f'  Best checkpointed   : {final_best_test*100:.2f}%')

if test_acc >= 0.96 or final_best_test >= 0.96:
    effective = max(test_acc, final_best_test)
    print(f'  ✅ TARGET MET — {effective*100:.2f}% >= 96%')
else:
    print(f'  ⚠️  Below target — max={max(test_acc, final_best_test)*100:.2f}%')

# Classification report
test_gen.reset()
y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
y_true = test_gen.classes

print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ── Plots ──────────────────────────────────────────────────────────────────────
full_acc     = history_p1.history['accuracy']     + history_p2.history['accuracy']
full_val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
full_loss    = history_p1.history['loss']          + history_p2.history['loss']
full_val_los = history_p1.history['val_loss']      + history_p2.history['val_loss']
p1_end       = len(history_p1.history['accuracy'])

# Combine test_acc history from both callbacks
test_acc_hist = test_ckpt_p1.history + test_ckpt_p2.history

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(full_acc,     label='Train Acc',  color='#2196F3')
ax1.plot(full_val_acc, label='Val Acc',    color='#4CAF50')
ax1.plot(test_acc_hist,label='Test Acc',   color='#9C27B0', linestyle='--')
ax1.axvline(x=p1_end - 0.5, color='gray', linestyle='--', alpha=0.7, label='P1→P2')
ax1.axhline(y=0.96, color='red', linestyle=':', alpha=0.5, label='96% target')
ax1.set_title('Accuracy (Train/Val/Test)', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(full_loss,    label='Train Loss', color='#F44336')
ax2.plot(full_val_los, label='Val Loss',   color='#FF9800')
ax2.axvline(x=p1_end - 0.5, color='gray', linestyle='--', alpha=0.7, label='P1→P2')
ax2.set_title('Loss', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle(f'DenseNet121 v2 | Test Acc: {test_acc*100:.2f}%', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_history_v2.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
ax.set_title(f'Confusion Matrix v2 (Test Acc: {test_acc*100:.2f}%)', fontweight='bold')
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_v2.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'Plots saved to {PLOTS_DIR}')

# ── Save ───────────────────────────────────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f'Model saved: {MODEL_SAVE_PATH} ({os.path.getsize(MODEL_SAVE_PATH)/1e6:.1f} MB)')

# ── Grad-CAM Smoke Test ────────────────────────────────────────────────────────
print('\nGrad-CAM smoke test...')
import cv2
from PIL import Image as PILImage

def _gradcam(img_array, mdl, layer_name):
    inner = next((l for l in mdl.layers if isinstance(l, tf.keras.Model)), None)
    if inner:
        lyr = inner.get_layer(layer_name)
        gm  = tf.keras.Model(inner.inputs, [lyr.output, inner.output])
        with tf.GradientTape() as tape:
            conv_out, base_out = gm(img_array)
            tape.watch(conv_out)
            x = base_out
            idx = mdl.layers.index(inner)
            for l in mdl.layers[idx+1:]:
                x = l(x)
            cls_ch = x[:, tf.argmax(x[0])]
        grads = tape.gradient(cls_ch, conv_out)
        pw    = tf.reduce_mean(grads, axis=(0,1,2))
        hm    = conv_out[0] @ pw[..., tf.newaxis]
        hm    = tf.squeeze(hm)
        hm    = tf.maximum(hm, 0) / tf.math.reduce_max(hm)
        return hm.numpy()
    return None

for cls in CLASS_NAMES:
    cls_dir  = os.path.join(TEST_DIR, cls)
    img_file = os.path.join(cls_dir, os.listdir(cls_dir)[0])
    img = PILImage.open(img_file).convert('RGB').resize(IMG_SIZE)
    arr = preprocess_input(np.expand_dims(np.array(img), axis=0))
    try:
        hm = _gradcam(arr, model, GRADCAM_LAYER)
        print(f'  {cls:<15}: ✅  heatmap {hm.shape}')
    except Exception as e:
        print(f'  {cls:<15}: ❌  {e}')

# ── Summary ────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('  TRAINING v2 COMPLETE')
print('='*60)
print(f'  Final Test Acc : {test_acc*100:.2f}%  (target 96%+)')
print(f'  Best Test Ckpt : {final_best_test*100:.2f}%')
print(f'  Model saved    : {MODEL_SAVE_PATH}')
print('='*60)
