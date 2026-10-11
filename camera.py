import requests
import os

CAMERA_URL = "http://192.168.4.1/capture"

SAVE_PATH = "captured/latest.jpg"


def capture_image():

    try:
        response = requests.get(
            CAMERA_URL,
            timeout=10
        )

        if response.status_code != 200:
            print("Camera error:", response.status_code)
            return False

        os.makedirs("captured", exist_ok=True)

        with open(SAVE_PATH, "wb") as file:
            file.write(response.content)

        print("Image received")
        print("Image size:", len(response.content), "bytes")

        return True

    except requests.exceptions.RequestException as e:

        print("Camera connection error:")
        print(e)

        return False