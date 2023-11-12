from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from gtts import gTTS
import re

app = Flask(__name__)

model_id = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(model_id)
processor = BlipProcessor.from_pretrained(model_id)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clean_caption(caption):
    # Remove unwanted characters, extra spaces, and numbers
    caption = re.sub(r'[^a-zA-Z.,!? ]', '', caption)
    return caption

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_url = request.form.get("image_url")
        image_file = request.files.get("image_file")
        language = request.form["language"]

        if image_url:
            image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        elif image_file:
            # Save the uploaded file to the uploads folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)
            image = Image.open(image_path).convert('RGB')
        else:
            # Handle the case where neither image URL nor file is provided
            return render_template("index.html")

        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        if language == "tagalog":
            # Translate the caption to Tagalog
            translator = Translator()
            caption = translator.translate(caption, src="en", dest="tl").text

        caption = clean_caption(caption)

        # Pass the image filename to the template
        return render_template("index.html", caption=caption, image_filename=image_file.filename)

    return render_template("index.html")

# Add this route at the end of your Flask code
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run()
