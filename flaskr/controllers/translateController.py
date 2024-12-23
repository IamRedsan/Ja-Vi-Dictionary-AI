from flask import request, Blueprint
from flaskr.errors.bad_request import BadRequestError
import numpy as np
import cv2
from flaskr.utils.image_to_text import image_to_text

translateBP = Blueprint("translate", __name__, url_prefix="/api/v1/translate")


@translateBP.post("")
def stranslate():
    data = request.json
    if not data:
        raise BadRequestError("Please fill the content to translate!")

    content = data.get("content")

    if not isinstance(content, str):
        raise BadRequestError("Only text allowed!")

    content = content.strip()

    if len(content) == 0:
        raise BadRequestError("Please fill the content to translate!")

    return {"msg": "Recieved request. Please wait"}


@translateBP.post("/image-to-text")
def imageToText():
    if "image" not in request.files or not request.files["image"].filename:
        raise BadRequestError("Missing image")

    uploaded_image = request.files["image"]
    format = uploaded_image.filename.rsplit(".", 1)[1].lower()
    if not format in ("png", "jpg", "jpeg"):
        raise BadRequestError("Unsupported image type")

    image_bytes = uploaded_image.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise BadRequestError("Failed to decode image")

    text = image_to_text(image)

    return {"text": text}
