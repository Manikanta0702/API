import os
from flask import Flask, request, jsonify
import openai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


openai.api_key = os.getenv('API_KEY')

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, 'rb') as audio_file:
            result = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
            transcription = result['text']
            return transcription.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def generate_reply(transcription):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides human-like responses. And give the response in marked down format."},
                {"role": "user", "content": transcription},
            ],
            temperature=0.7,
            max_tokens=100,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}" 

@app.route("/process-audio/", methods=['POST', 'GET'])
def process_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        audio_file_path = f"temp_{file.filename}"
        file.save(audio_file_path)

        transcription = transcribe_audio(audio_file_path)

        if "Error" in transcription:
            os.remove(audio_file_path)
            return jsonify({"error": transcription}), 400

        response = generate_reply(transcription)

        os.remove(audio_file_path)  
        return jsonify({"response": response, "trancsript": transcription})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
