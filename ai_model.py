# ai_model.py
# ============================================================
# ReactX AI - Plastic Classification
# ============================================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = r"D:\Chemovista\model\chemovista_resnet18.pth"

# ============================================================
# CLASS NAMES
# Must match the order used during training
# ============================================================

CLASS_NAMES = [
    "HDPE",
    "OTHER",
    "PET",
    "PP",
    "PVC"
]

# ============================================================
# CONFIDENCE THRESHOLD
# ============================================================

CONFIDENCE_THRESHOLD = 70.0

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

# ============================================================
# LOAD MODEL
# ============================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support either a complete state_dict
# or a checkpoint containing "model_state_dict"

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(DEVICE)

model.eval()

print("================================")
print("ReactX AI model loaded")
print("Classes:", CLASS_NAMES)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("================================")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_plastic(image_path):

    try:

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        input_tensor = transform(
            image
        )

        input_batch = input_tensor.unsqueeze(0)

        input_batch = input_batch.to(
            DEVICE
        )

        # ----------------------------------------------------
        # AI PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_batch
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

        # ----------------------------------------------------
        # FIND BEST CLASS
        # ----------------------------------------------------

        confidence_tensor, class_index_tensor = torch.max(
            probabilities,
            dim=1
        )

        confidence = (
            confidence_tensor.item()
            * 100
        )

        class_index = (
            class_index_tensor.item()
        )

        predicted_class = (
            CLASS_NAMES[class_index]
        )

        # ----------------------------------------------------
        # CONFIDENCE FILTER
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            print(
                f"Low confidence: "
                f"{confidence:.2f}%"
            )

            print(
                "Result changed to OTHER"
            )

            predicted_class = "OTHER"

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            "AI Prediction:",
            predicted_class
        )

        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return (
            predicted_class,
            confidence
        )

    except Exception as e:

        print(
            "AI prediction error:"
        )

        print(e)

        return (
            "OTHER",
            0.0
        )