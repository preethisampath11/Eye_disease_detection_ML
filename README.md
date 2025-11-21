Concept of the Model

This project uses MobileNetV2 as the backbone for multi-label eye disease detection.
MobileNetV2 is a lightweight CNN pretrained on ImageNet, used here as a feature extractor.
The pretrained layers are frozen, and custom dense layers are added on top to learn disease-specific patterns.

The model takes retinal images as input, extracts deep visual features using MobileNetV2, and then passes them through fully connected layers to predict the presence of multiple diseases simultaneously.
The final layer uses sigmoid activation, allowing the model to assign independent probabilities for each disease.

Image augmentation and MobileNetV2 preprocessing are used to improve generalization.
The model trains using binary cross-entropy, with accuracy and AUC as metrics.

In summary:
MobileNetV2 + custom classification layers → multi-label disease prediction with efficient and transferable feature learning.
