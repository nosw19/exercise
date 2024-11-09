from flask import Flask, request, render_template_string, Response
import tempfile
import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# HTML 템플릿
HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>운동 동작 실시간 분석</title>
  <style>
    body {
      font-family: 'Nanum Gothic', sans-serif;
      background-color: #f0f8ff;
      color: #333333;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    h1 {
      color: #2c3e50;
      font-size: 2.5em;
      margin-top: 20px;
    }
    .form-container {
      display: flex;
      gap: 20px;
      margin-top: 20px;
    }
    .form-box {
      background-color: #ffffff;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 280px;
    }
    label {
      font-size: 1.1em;
      margin-bottom: 10px;
    }
    input[type="file"], input[type="submit"] {
      font-size: 1em;
    }
    input[type="submit"] {
      background-color: #3498db;
      color: white;
      font-size: 1.1em;
      padding: 10px 20px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      margin-top: 10px;
    }
    input[type="submit"]:hover {
      background-color: #2980b9;
    }
  </style>
</head>
<body>
  <h1>운동 동작 실시간 분석</h1>

  <div class="form-container">
    <!-- 동영상 분석 네모 박스 -->
    <div class="form-box">
      <h2>동영상 분석</h2>
      <form action="/start_video" method="post" enctype="multipart/form-data">
        <label>모델 파일 선택:</label>
        <input type="file" name="model" required>
        <label>동영상 파일 선택:</label>
        <input type="file" name="video" required>
        <input type="submit" value="동영상 분석 시작">
      </form>
    </div>

    <!-- 실시간 분석 네모 박스 -->
    <div class="form-box">
      <h2>실시간 분석</h2>
      <form action="/start_stream" method="post" enctype="multipart/form-data">
        <label>모델 파일 선택:</label>
        <input type="file" name="model" required>
        <input type="submit" value="실시간 분석 시작">
      </form>
    </div>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

def load_keras_model(file_path):
    """Keras 모델을 안전하게 불러오는 함수"""
    try:
        model = tf.keras.models.load_model(file_path)
        return model
    except Exception as e:
        print(f"Keras 모델 로드 시 오류 발생: {e}")
        return None

def predict_exercise(model, frame_sequence):
    sequence = np.array(frame_sequence).reshape(1, len(frame_sequence), 1)
    prediction = model.predict(sequence)
    exercise_idx = np.argmax(prediction)
    return unique_labels[exercise_idx]

unique_labels = ['스텝 백워드 다이나믹 런지', '스탠딩 니업', '바벨 로우', '버피 테스트', '플랭크', 
                 '시저크로스', '힙쓰러스트', '푸시업', '업라이트로우', '스텝 포워드 다이나믹 런지']

@app.route("/start_video", methods=["POST"])
def process_video():
    model_file = request.files.get("model")
    video_file = request.files.get("video")

    if not model_file or not video_file:
        return "모델 파일과 동영상 파일을 모두 업로드해야 합니다.", 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model:
        temp_model.write(model_file.read())
        model_path = temp_model.name

    classification_model = load_keras_model(model_path)
    if classification_model is None:
        return "모델을 불러오는 중 오류가 발생했습니다.", 500

    video_path = f"./static/{video_file.filename}"
    video_file.save(video_path)

    return Response(analyze_video(video_path, classification_model), mimetype='multipart/x-mixed-replace; boundary=frame')

def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = deque(maxlen=17)  # 프레임 길이 설정

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_sequence.append(np.mean(frame))  # 프레임을 평균하여 시퀀스에 추가
        if len(frame_sequence) == 17:
            exercise = predict_exercise(model, frame_sequence)
            frame = draw_text_korean(frame, exercise, (10, 50))

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/start_stream", methods=["POST"])
def start_stream():
    model_file = request.files.get("model")

    if not model_file:
        return "모델 파일은 필수입니다.", 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model:
        temp_model.write(model_file.read())
        model_path = temp_model.name

    classification_model = load_keras_model(model_path)
    if classification_model is None:
        return "모델을 불러오는 중 오류가 발생했습니다.", 500

    return Response(analyze_stream(classification_model), mimetype='multipart/x-mixed-replace; boundary=frame')

def analyze_stream(model):
    cap = cv2.VideoCapture(0)
    frame_sequence = deque(maxlen=17)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_sequence.append(np.mean(frame))
        if len(frame_sequence) == 17:
            exercise = predict_exercise(model, frame_sequence)
            frame = draw_text_korean(frame, exercise, (10, 50))

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def draw_text_korean(image, text, position, font_path='NanumGothic.ttf', font_size=30, color=(255, 255, 255)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
