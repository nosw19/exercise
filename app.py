import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# 모델 로드 함수
def load_keras_model(file_path):
    """Keras 모델을 안전하게 불러오는 함수"""
    try:
        model = tf.keras.models.load_model(file_path)
        return model
    except Exception as e:
        st.error(f"Keras 모델 로드 시 오류 발생: {e}")
        return None

# 예측 함수
def predict_exercise(model, frame_sequence):
    sequence = np.array(frame_sequence).reshape(1, len(frame_sequence), 1)
    prediction = model.predict(sequence)
    exercise_idx = np.argmax(prediction)
    return unique_labels[exercise_idx]

# 한국어 텍스트 그리기 함수
def draw_text_korean(image, text, position, font_path='NanumGothic.ttf', font_size=30, color=(255, 255, 255)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

unique_labels = ['스텝 백워드 다이나믹 런지', '스탠딩 니업', '바벨 로우', '버피 테스트', '플랭크', 
                 '시저크로스', '힙쓰러스트', '푸시업', '업라이트로우', '스텝 포워드 다이나믹 런지']

st.title("운동 동작 실시간 분석")

# 모델 파일 업로드
model_file = st.file_uploader("모델 파일 선택", type=["keras"])
video_file = st.file_uploader("동영상 파일 선택 (선택사항)", type=["mp4", "avi"])

if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model:
        temp_model.write(model_file.read())
        model_path = temp_model.name

    classification_model = load_keras_model(model_path)
    if classification_model is not None:
        st.success("모델이 성공적으로 로드되었습니다.")

        if video_file:
            st.video(video_file)

            # 비디오 분석
            cap = cv2.VideoCapture(video_file.name)
            frame_sequence = deque(maxlen=17)
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_sequence.append(np.mean(frame))
                if len(frame_sequence) == 17:
                    exercise = predict_exercise(classification_model, frame_sequence)
                    frame = draw_text_korean(frame, exercise, (10, 50))
                    frames.append(frame)

            cap.release()

            # 비디오 스트림 표시
            st.write("비디오 분석 결과:")
            for frame in frames:
                st.image(frame, channels="BGR")

        else:
            st.write("실시간 분석을 위해 웹캠을 선택하세요.")

            # 실시간 분석
            if st.button("실시간 분석 시작"):
                cap = cv2.VideoCapture(0)
                frame_sequence = deque(maxlen=17)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_sequence.append(np.mean(frame))
                    if len(frame_sequence) == 17:
                        exercise = predict_exercise(classification_model, frame_sequence)
                        frame = draw_text_korean(frame, exercise, (10, 50))
                        st.image(frame, channels="BGR")
                        
                cap.release()
