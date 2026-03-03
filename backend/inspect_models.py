import tensorflow as tf

effnet = tf.keras.models.load_model(r"c:\Users\Siddharth Gupta\Desktop\main_encephlo\Encephlo\backend\models\efficientnetb0.keras")
densenet = tf.keras.models.load_model(r"c:\Users\Siddharth Gupta\Desktop\main_encephlo\Encephlo\backend\models\densenet121.keras")

print("EFFICIENTNET SUMMARY:")
effnet.summary()

print("DENSENET SUMMARY:")
densenet.summary()
