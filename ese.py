from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

# Replace with your actual endpoint and API key
endpoint = "https://lakshyavarshney.cognitiveservices.azure.com/"
api_key = "948c1c7dc9934a2bb2c661ed0bd6786e"
# Create a ComputerVisionClient instance
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))

# Test API call (replace with your actual image URL)
image_url = "temp.jpg"

try:
    # Perform OCR on the image
    ocr_result = computervision_client.recognize_printed_text(image_url)

    # Print OCR result
    if ocr_result.status_code == OperationStatusCodes.succeeded:
        for region in ocr_result.regions:
            for line in region.lines:
                print(' '.join([word.text for word in line.words]))
    else:
        print("OCR operation failed with status code:", ocr_result.status_code)

except Exception as e:
    print("Error:", str(e))
