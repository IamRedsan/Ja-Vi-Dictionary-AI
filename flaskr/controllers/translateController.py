from flask import request, Blueprint
from flaskr.errors.bad_request import BadRequestError
from io import BytesIO
from PIL import Image
import cloudinary.uploader
from flaskr.utils.image_to_text import (
    image_to_text,
    translate_pars_v1,
    translate_pars_v2,
)

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


@translateBP.post("/from-image-v1")
def fromImageV1():
    if "image" not in request.files or not request.files["image"].filename:
        raise BadRequestError("Missing image")

    image = request.files["image"]
    format = image.filename.rsplit(".", 1)[1].lower()
    if not format in ("png", "jpg", "jpeg"):
        raise BadRequestError("Unsupported image type")

    img = Image.open(BytesIO(image.read()))

    paragraphs = image_to_text(img)
    result = translate_pars_v1(img, paragraphs)

    if format == "png":
        img = img.convert("RGBA")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
    else:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    cloudinaryResponse = cloudinary.uploader.upload(img_byte_arr, resource_type="image")

    return {"imageUrl": cloudinaryResponse["secure_url"], "result": result}


@translateBP.post("/from-image-v2")
def fromImageV2():
    if "image" not in request.files or not request.files["image"].filename:
        raise BadRequestError("Missing image")

    image = request.files["image"]
    format = image.filename.rsplit(".", 1)[1].lower()
    if not format in ("png", "jpg", "jpeg"):
        raise BadRequestError("Unsupported image type")

    img = Image.open(BytesIO(image.read()))

    paragraphs = image_to_text(img)
    result = translate_pars_v2(img, paragraphs)

    if format == "png":
        img = img.convert("RGBA")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
    else:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    cloudinaryResponse = cloudinary.uploader.upload(img_byte_arr, resource_type="image")

    return {"imageUrl": cloudinaryResponse["secure_url"], "result": result}
