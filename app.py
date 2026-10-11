from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory
)

import os
import uuid
import requests

from ai_model import predict_plastic


# ============================================================
# REACTX AI
# ============================================================

app = Flask(__name__)


# ============================================================
# FOLDER
# ============================================================

CAPTURE_FOLDER = "captured"

os.makedirs(
    CAPTURE_FOLDER,
    exist_ok=True
)


# ============================================================
# ESP32 IP ADDRESSES
# ============================================================

# ESP32-CAM
ESP32_CAM_IP = "192.168.9.182"

CAMERA_URL = (
    f"http://{ESP32_CAM_IP}/image"
)


# ESP32 DEV MODULE
ESP32_DEV_IP = "192.168.9.158"


# ============================================================
# BELT STATE
# ============================================================

belt_state = "OFF"


# ============================================================
# LATEST AI RESULT
# ============================================================

latest_data = {

    "success": False,

    "plastic": "",

    "confidence": 0,

    "image": "",

    "status": "Waiting for object..."

}


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# MONITOR
# ============================================================

@app.route("/monitor")
def monitor():

    return render_template(
        "monitor.html"
    )


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    return jsonify({

        "success": True,

        "camera_ip":
            ESP32_CAM_IP,

        "esp32_ip":
            ESP32_DEV_IP,

        "belt":
            belt_state,

        "latest":
            latest_data

    })


# ============================================================
# LATEST RESULT
# ============================================================

@app.route("/latest_result")
def latest_result():

    return jsonify(
        latest_data
    )


# ============================================================
# CAPTURED IMAGES
# ============================================================

@app.route("/captured/<filename>")
def captured_file(filename):

    return send_from_directory(
        CAPTURE_FOLDER,
        filename
    )



# ============================================================
# DIRECT ESP32-CAM UPLOAD
# ============================================================

@app.route("/camera", methods=["POST"])
def camera_upload():

    global latest_data

    try:

        print()
        print("=" * 50)
        print("REACTX AI - DIRECT CAMERA UPLOAD")
        print("=" * 50)

        if "image" not in request.files:
            latest_data = {
                "success": False,
                "plastic": "",
                "confidence": 0,
                "image": "",
                "status": "No image received from ESP32-CAM"
            }

            return jsonify(latest_data), 400

        file = request.files["image"]

        if file.filename == "":
            latest_data = {
                "success": False,
                "plastic": "",
                "confidence": 0,
                "image": "",
                "status": "Empty camera filename"
            }

            return jsonify(latest_data), 400

        filename = (
            "capture_"
            + uuid.uuid4().hex
            + ".jpg"
        )

        filepath = os.path.join(
            CAPTURE_FOLDER,
            filename
        )

        file.save(filepath)

        print(
            "Image received from ESP32-CAM:",
            os.path.getsize(filepath),
            "bytes"
        )

        # Show the captured image immediately while AI runs.
        latest_data = {
            "success": False,
            "plastic": "",
            "confidence": 0,
            "image": "/captured/" + filename,
            "status": "AI analyzing image..."
        }

        plastic_type, confidence = predict_plastic(filepath)

        latest_data = {
            "success": True,
            "plastic": plastic_type,
            "confidence": round(float(confidence), 2),
            "image": "/captured/" + filename,
            "status": "AI analysis complete"
        }

        print("Plastic:", plastic_type)
        print("Confidence:", confidence)
        print("Image:", filepath)

        return jsonify(latest_data)

    except Exception as e:

        print("Direct camera error:", str(e))

        latest_data = {
            "success": False,
            "plastic": "",
            "confidence": 0,
            "image": latest_data.get("image", ""),
            "status": "Camera/AI error: " + str(e)
        }

        return jsonify(latest_data), 500


# ============================================================
# PROCESS ESP32-CAM
# ============================================================

@app.route("/process_camera")
def process_camera():

    global latest_data

    # ========================================================
    # SAFE CAMERA TRIGGER
    # ========================================================
    #
    # ESP32-CAM /capture performs the complete single-capture
    # pipeline:
    #
    #   CAM captures ONE image
    #       -> POST /camera
    #       -> AI prediction
    #       -> CAM receives result
    #       -> this route receives that same result
    #
    # IMPORTANT:
    # Do NOT request /image here. That would create a second
    # camera capture when the Dev Module has already triggered
    # /capture.
    # ========================================================

    try:
        print()
        print("=" * 50)
        print("REACTX AI - SAFE CAMERA TRIGGER")
        print("=" * 50)

        trigger_url = (
            f"http://{ESP32_CAM_IP}/capture?t={uuid.uuid4().int}"
        )

        print("Requesting:", trigger_url)

        response = requests.get(
            trigger_url,
            timeout=45
        )

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            raise Exception(
                "ESP32-CAM returned a non-JSON response: "
                + response.text[:300]
            )

        latest_data = data

        print()
        print("CAMERA RESULT")
        print("Plastic:", data.get("plastic", ""))
        print("Confidence:", data.get("confidence", ""))
        print("Image:", data.get("image", ""))
        print("Status:", data.get("status", ""))

        return jsonify(latest_data)

    except requests.exceptions.Timeout:
        latest_data = {
            "success": False,
            "plastic": "",
            "confidence": 0,
            "image": latest_data.get("image", ""),
            "status": "Camera request timed out"
        }

        print("Camera trigger timeout")
        return jsonify(latest_data), 504

    except requests.exceptions.ConnectionError:
        latest_data = {
            "success": False,
            "plastic": "",
            "confidence": 0,
            "image": latest_data.get("image", ""),
            "status": (
                "Cannot connect to ESP32-CAM at "
                + ESP32_CAM_IP
            )
        }

        print("Cannot connect to ESP32-CAM")
        return jsonify(latest_data), 503

    except Exception as e:
        print("Camera/AI error:", str(e))

        latest_data = {
            "success": False,
            "plastic": "",
            "confidence": 0,
            "image": latest_data.get("image", ""),
            "status": "Camera/AI error: " + str(e)
        }

        return jsonify(latest_data), 500


# ============================================================
# MANUAL PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    global latest_data


    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image uploaded"

        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No image selected"

        }), 400


    try:

        extension = ".jpg"


        if "." in file.filename:

            extension = (

                "."

                + file.filename
                .rsplit(
                    ".",
                    1
                )[1]

            )


        filename = (

            "upload_"

            + uuid.uuid4().hex

            + extension

        )


        filepath = os.path.join(

            CAPTURE_FOLDER,

            filename

        )


        file.save(
            filepath
        )


        plastic_type, confidence = (
            predict_plastic(filepath)
        )


        latest_data = {

            "success": True,

            "plastic":
                plastic_type,

            "confidence":
                round(
                    float(confidence),
                    2
                ),

            "image":
                "/captured/"
                + filename
                + "?t="
                + str(uuid.uuid4().int),

            "status":
                "AI analysis complete"

        }


        return jsonify(
            latest_data
        )


    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# BELT CONTROL
# ============================================================

@app.route("/belt_control")
def belt_control():

    global belt_state


    action = request.args.get(
        "action",
        ""
    ).lower().strip()


    print()
    print("=" * 50)
    print("REACTX AI - BELT CONTROL")
    print("=" * 50)

    print(
        "Command:",
        action
    )

    print(
        "ESP32:",
        ESP32_DEV_IP
    )


    # ========================================================
    # BELT ON
    # ========================================================

    if action == "on":

        url = (
            f"http://{ESP32_DEV_IP}/belt/on"
        )


        try:

            print(
                "Sending:",
                url
            )


            response = requests.get(

                url,

                timeout=5

            )


            print(
                "ESP32 response:",
                response.text
            )


            response.raise_for_status()


            belt_state = "ON"


            return jsonify({

                "success": True,

                "belt": "ON",

                "message":
                    "Belt turned ON",

                "esp32_response":
                    response.text

            })


        except requests.exceptions.Timeout:

            print(
                "ERROR: ESP32 connection timeout"
            )


            return jsonify({

                "success": False,

                "belt": belt_state,

                "error":
                    "ESP32 Dev Module did not respond. "
                    "Check IP, Wi-Fi and ESP32 server."

            }), 504


        except requests.exceptions.ConnectionError:

            print(
                "ERROR: Cannot connect to ESP32"
            )


            return jsonify({

                "success": False,

                "belt": belt_state,

                "error":
                    "Cannot connect to ESP32 at "
                    + ESP32_DEV_IP

            }), 503


        except Exception as e:

            print(
                "Belt ON error:",
                str(e)
            )


            return jsonify({

                "success": False,

                "belt": belt_state,

                "error":
                    str(e)

            }), 500


    # ========================================================
    # BELT OFF
    # ========================================================

    elif action == "off":

        url = (
            f"http://{ESP32_DEV_IP}/belt/off"
        )


        try:

            print(
                "Sending:",
                url
            )


            response = requests.get(

                url,

                timeout=5

            )


            print(
                "ESP32 response:",
                response.text
            )


            response.raise_for_status()


            belt_state = "OFF"


            return jsonify({

                "success": True,

                "belt": "OFF",

                "message":
                    "Belt turned OFF",

                "esp32_response":
                    response.text

            })


        except requests.exceptions.Timeout:

            print(
                "ERROR: ESP32 connection timeout"
            )


            return jsonify({

                "success": False,

                "belt": belt_state,

                "error":
                    "ESP32 Dev Module did not respond."

            }), 504


        except requests.exceptions.ConnectionError:

            return jsonify({

                "success": False,

                "belt": belt_state,

                "error":
                    "Cannot connect to ESP32 at "
                    + ESP32_DEV_IP

            }), 503


        except Exception as e:

            return jsonify({

                "success": False,

                "belt": belt_state,

                "error":
                    str(e)

            }), 500


    # ========================================================
    # INVALID COMMAND
    # ========================================================

    else:

        return jsonify({

            "success": False,

            "error":
                "Invalid command. "
                "Use on or off."

        }), 400


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("                 REACTX AI")
    print("        SMART PLASTIC SORTING SYSTEM")
    print("=" * 60)

    print()

    print(
        "Laptop / Flask : 192.168.9.110"
    )

    print(
        "ESP32-CAM      :",
        ESP32_CAM_IP
    )

    print(
        "ESP32 Dev      :",
        ESP32_DEV_IP
    )

    print()

    print(
        "ReactX AI:"
    )

    print(
        "Single capture flow: ESP32-CAM /capture -> Flask /camera"
    )

    print(
        "http://192.168.9.110:5000"
    )

    print()

    print(
        "Belt ON:"
    )

    print(
        f"http://{ESP32_DEV_IP}/belt/on"
    )

    print()

    print(
        "Belt OFF:"
    )

    print(
        f"http://{ESP32_DEV_IP}/belt/off"
    )

    print()
    print("=" * 60)


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )