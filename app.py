import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# Define unique labels
unique_labels = ['스텝 백워드 다이나믹 런지', '스탠딩 니업', '바벨 로우', '버피 테스트', '플랭크', 
                 '시저크로스', '힙쓰러스트', '푸시업', '업라이트로우', '스텝 포워드 다이나믹 런지']

# Function to load Keras model
def load_keras_model(file_path):
    try:
        model = tf.keras.models.load_model(file_path)
        return model
    except Exception as e:
        st.error(f"Keras 모델 로드 시 오류 발생: {e}")
        return None

# Predict exercise based on frame sequence
def predict_exercise(model, frame_sequence):
    sequence = np.array(frame_sequence).reshape(1, len(frame_sequence), 1)
    prediction = model.predict(sequence)
    exercise_idx = np.argmax(prediction)
    return unique_labels[exercise_idx]

# Draw Korean text on image
def draw_text_korean(image, text, position, font_path='C:/Users/itwill/Desktop/나눔 글꼴/나눔고딕/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf', font_size=30, color=(255, 255, 255)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# Analyze video
def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = deque(maxlen=17)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_sequence.append(np.mean(frame))
        if len(frame_sequence) == 17:
            exercise = predict_exercise(model, frame_sequence)
            frame = draw_text_korean(frame, exercise, (10, 50))

        st.image(frame, channels="BGR")
    cap.release()

# Analyze stream from webcam
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

        st.image(frame, channels="BGR")
    cap.release()

# Streamlit UI
st.title("운동 동작 실시간 분석")

# File upload for Keras model
model_file = st.file_uploader("모델 파일을 선택하세요 (.keras)", type="keras")
video_file = st.file_uploader("동영상 파일을 선택하세요", type=["mp4", "avi"])

if model_file:
    # Save uploaded model to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model:
        temp_model.write(model_file.read())
        model_path = temp_model.name

    classification_model = load_keras_model(model_path)

    if classification_model:
        # Options for Video or Webcam Stream
        option = st.radio("분석 방법 선택", ("동영상 파일", "실시간 웹캠"))
        
        if option == "동영상 파일" and video_file:
            # Save uploaded video file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name
            
            st.write("동영상 분석 시작")
            analyze_video(video_path, classification_model)
        
        elif option == "실시간 웹캠":
            st.write("실시간 웹캠 분석 시작")
            analyze_stream(classification_model)
    else:
        st.error("모델을 불러오는 중 오류가 발생했습니다.")
