from flask import request, Blueprint
from flaskr.errors.bad_request import BadRequestError

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
