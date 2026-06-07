# Trained model weights

Drop the three Keras `.h5` files here before building the Docker image:

```
models/
├── resnet_model.h5
├── densenet_model.h5
└── efficientnet_model.h5
```

Each model must:
- accept input shape `(224, 224, 3)`
- output **5 softmax probabilities** in this exact order:
  `["Normal", "Glaucoma", "Diabetic Retinopathy", "Cataract", "AMD"]`
- have been trained with the standard backbone preprocessing
  (`tensorflow.keras.applications.<backbone>.preprocess_input`)

If a file is missing, the backend falls back to an ImageNet-initialised
backbone with a random classifier head (clearly marked in the logs) so the
API stays callable. **This is not a clinical fallback** — replace the files
for real inference.