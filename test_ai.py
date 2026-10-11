from ai_model import predict_plastic


IMAGE_PATH = r"D:\Chemovista\Chemovista_Dataset\test\PET"


# Find the first image in the PET test folder
import os

files = os.listdir(IMAGE_PATH)

image_file = None

for file in files:

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):

        image_file = file
        break


if image_file is None:

    print("No image found.")

else:

    full_path = os.path.join(
        IMAGE_PATH,
        image_file
    )

    print()
    print("Testing image:")
    print(full_path)
    print()

    plastic, confidence = predict_plastic(
        full_path
    )

    print("================================")
    print("CHEMOVISTA PREDICTION")
    print("================================")
    print()

    print(
        f"Plastic: {plastic}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print()