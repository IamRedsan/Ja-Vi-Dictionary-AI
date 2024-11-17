import pytesseract
from flaskr.utils.image_utils import ImageText
from PIL import ImageDraw, ImageFont

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def image_to_text(image):

    data = pytesseract.image_to_data(
        image=image,
        lang="jpn+jpn_vert",
        output_type=pytesseract.Output.DICT,
    )

    data_len = len(data["conf"])

    paragraphs = dict()

    for i in range(data_len):

        if data["conf"][i] == -1:
            continue

        if not data["text"][i].strip():
            continue

        key = f'{data["block_num"][i]}-{data["par_num"][i]}'

        paragraph = paragraphs.get(key)

        if not paragraph:
            paragraph = dict(
                {
                    "text": "",
                    "left": 10**6,
                    "right": -1,
                    "top": 10**6,
                    "bottom": -1,
                    "fontSize": -1,
                }
            )

            paragraphs[key] = paragraph

        paragraph["text"] = paragraph["text"] + data["text"][i]
        paragraph["left"] = min(paragraph["left"], data["left"][i])
        paragraph["right"] = max(paragraph["right"], data["left"][i] + data["width"][i])
        paragraph["top"] = min(paragraph["top"], data["top"][i])
        paragraph["bottom"] = max(
            paragraph["bottom"], data["top"][i] + data["height"][i]
        )
        paragraph["fontSize"] = max(
            paragraph["fontSize"], min(data["height"][i], data["width"][i])
        )

    return paragraphs


from googletrans import Translator

translator = Translator()


def translate(input_text):
    return translator.translate(input_text, dest="vi").text


def translate_pars_v1(image, paragraphs):

    img = ImageText(image)
    font = "arial.ttf"
    color = (50, 50, 50)

    result = []
    for paragraph in paragraphs.values():
        translated_text = translate(paragraph["text"])
        paragraph["translated_text"] = translated_text
        result.append(paragraph)

        x, y = paragraph["left"], paragraph["top"]
        w = paragraph["right"] - paragraph["left"]
        h = paragraph["bottom"] - paragraph["top"]
        font_size = paragraph["fontSize"] // 2

        img.fill_rectangle((x, y), (x + w, y + h), (0, 255, 255))

        img.write_text_box(
            (x, y),
            translated_text,
            box_width=w,
            font_filename=font,
            font_size=font_size,
            color=color,
        )
    return result


colors = ["red", "green", "blue", "orange", "purple"]


def translate_pars_v2(image, paragraphs):

    draw = ImageDraw.Draw(image)

    result = []
    i = 0
    for paragraph in paragraphs.values():
        translated_text = translate(paragraph["text"])
        paragraph["translated_text"] = translated_text
        paragraph["number"] = i + 1
        result.append(paragraph)

        top = paragraph["top"]
        left = paragraph["left"]
        bottom = paragraph["bottom"]
        right = paragraph["right"]

        color = colors[i % len(colors)]

        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        font_path = "arial.ttf"
        font_size = paragraph["fontSize"]
        font = ImageFont.truetype(font_path, font_size)
        draw.text((left, top), str(i + 1), fill=color, font=font)

        i += 1
    return result
