from tensorflow.keras.models import load_model

# Load using the correct file name
model = load_model("model/unet.h5", compile=False)

# Print model summary
model.summary()

