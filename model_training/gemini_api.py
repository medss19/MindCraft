import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
from google.cloud import vision


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def image_to_text(path):
    client = vision.ImageAnnotatorClient.from_service_account_json('path/to/your/google-cloud-vision-api-key.json')
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    text = response.text_annotations[0].description
    return text