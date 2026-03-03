I checked your local environment, and it seems **TensorFlow is not picking up a GPU/CUDA** on your machine. All calculations are currently defaulting to the CPU.

Because a Vision Transformer (`vit-base`) is extremely heavy and we are fine-tuning it:

*   **Training time strictly on your CPU:** It's taking roughly **45-60 seconds per training step**. With 357 steps per epoch, that equates to about **4-5 hours per epoch**. For 10 epochs, it would take **40 to 50 hours** to finish Step 1 locally.

**How to fix this quickly:**
If you don't have a dedicated local GPU (like an Nvidia RTX) or CUDA isn't configured, I **highly** recommend you and Aditya:
1. Upload the `MRI images` folder and `train_vit.py` to **Google Colab** (it's free).
2. Set the Runtime to **T4 GPU** on Colab.
3. Run the script there—it will probably take **less than 15-20 minutes** on a cloud GPU!
4. Download the `vit_feature_extractor` folder back to your local `models/` directory and proceed to Step 2 locally.
