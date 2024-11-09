import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import os

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
def draw_text_korean(image, text, position, font_size=30, color=(255, 255, 255)):
    font_path = "NanumGothic.ttf"  # 현재 디렉터리에 위치한 폰트 파일 경로
    if not os.path.exists(font_path):
        st.error("폰트 파일을 찾을 수 없습니다. `NanumGothic.ttf` 파일을 업로드하세요.")
        return image
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
            # 비디오 분석
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name

            cap = cv2.VideoCapture(video_path)
            frame_sequence = deque(maxlen=17)
            placeholder = st.empty()  # Streamlit에서 프레임을 업데이트할 위치 지정

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 프레임을 회색조로 변환하여 평균 값을 시퀀스에 추가
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_sequence.append(np.mean(frame_gray))

                # 프레임 시퀀스가 원하는 길이에 도달하면 예측 수행
                if len(frame_sequence) == 17:
                    exercise = predict_exercise(classification_model, frame_sequence)
                    frame = draw_text_korean(frame, exercise, (10, 50))
                    placeholder.image(frame, channels="BGR")  # 프레임을 갱신

            cap.release()

        else:
            st.write("동영상 파일을 업로드하면 분석 결과가 표시됩니다.")
