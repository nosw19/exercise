import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import tensorflow as tf
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import ffmpeg

# 모델 로드 함수
def load_keras_model(file_path):
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

# 한국어 텍스트 표시 함수
def draw_text_korean(image, text, position, font_path='NanumGothic.ttf', font_size=30, color=(255, 255, 255)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# 라벨 정의
unique_labels = ['스텝 백워드 다이나믹 런지', '스탠딩 니업', '바벨 로우', '버피 테스트', '플랭크', 
                 '시저크로스', '힙쓰러스트', '푸시업', '업라이트로우', '스텝 포워드 다이나믹 런지']

st.title("운동 동작 분석 및 결과 비디오 생성")

# 모델 파일 및 비디오 파일 업로드
model_file = st.file_uploader("모델 파일 선택", type=["keras"])
video_file = st.file_uploader("분석할 비디오 파일 선택", type=["mp4", "avi"])

if model_file and video_file:
    # 임시 파일로 모델 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model:
        temp_model.write(model_file.read())
        model_path = temp_model.name

    classification_model = load_keras_model(model_path)
    if classification_model is not None:
        st.success("모델이 성공적으로 로드되었습니다.")

        # 임시 파일로 비디오 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        # 비디오 캡처 및 분석 비디오 생성
        cap = cv2.VideoCapture(video_path)
        frame_sequence = deque(maxlen=17)
        
        # 비디오 저장 설정
        output_path = os.path.join(tempfile.gettempdir(), "analyzed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 시퀀스 업데이트 및 예측
            frame_sequence.append(np.mean(frame))
            if len(frame_sequence) == 17:
                exercise = predict_exercise(classification_model, frame_sequence)
                frame = draw_text_korean(frame, exercise, (10, 50))

            # 결과가 포함된 프레임을 비디오에 저장
            out.write(frame)

        # 비디오 및 저장 파일 닫기
        cap.release()
        out.release()

        # FFmpeg를 사용하여 형식 변환
        converted_path = os.path.join(tempfile.gettempdir(), "converted_analyzed_video.mp4")
        ffmpeg.input(output_path).output(converted_path).run(overwrite_output=True)

        # 변환된 비디오 파일 표시
        st.video(converted_path)
