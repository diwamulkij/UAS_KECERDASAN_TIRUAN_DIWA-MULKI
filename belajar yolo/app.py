#import library yang dibutuhkan
from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
import base64

# ini untuk Muat model YOLO
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ini code untuk Memproses dan menampilkan hasil deteksi
def display_results(image, results):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Skor Tingkat keyakinannya
    labels = results.boxes.cls.cpu().numpy()  # Indeks kelas
    names = results.names  # nama kelas
    
    detected_objects = []
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Ambang batas kepercayaan
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            detected_objects.append(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detected_objects

# Fungsi untuk menambahkan latar belakang
def set_background(image_path):
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
# Fungsi untuk menampilkan GIF di sidebar
def display_animation(gif_path):
    with open(gif_path, "rb") as file:
        gif_data = file.read()
    base64_gif = base64.b64encode(gif_data).decode()
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/gif;base64,{base64_gif}" alt="Animation" width="200">
        </div>
        """,
        unsafe_allow_html=True,
    )

# ini untuk App Streamlit Utama
def main():
    # Tampilkan animasi di bawah sidebar
    display_animation("bumibg.gif")  # Path ke file GIF
    
    # ini code menambahkan image(Foto)
    set_background("planet-earth-background.jpg")  # URL ke gambar latar belakang

    st.title(" DETEKSI OBJEK DIWA MULKI ")
    st.sidebar.title("MENU")
    
    model_path = "yolo11n.pt"  # Jalur menyambungkan ke model YOLO kite
    model = load_model(model_path)

    # Buat tombol sakelar untuk mode gaming
    if "detection_active" not in st.session_state:
        st.session_state.detection_active = False

    if st.sidebar.button(" MULAI DETEKSI " if not st.session_state.detection_active else "❌ STOP DETEKSI ❌"):
        st.session_state.detection_active = not st.session_state.detection_active

    # Buka pengambilan video jika mode gaming aktif
    if st.session_state.detection_active:
        st.sidebar.success("MENYALAH CUY: 🟢 Active")
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()  # Placeholder untuk bingkai video
        st_detection_info = st.empty()  # Placeholder untuk informasi deteksi

        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture image.")
                break

            # Jalankan deteksi YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversikan ke RGB untuk tampilan
            results = model.predict(frame, imgsz=640)  # Lakukan deteksi
            
            # ini Gambarkan hasil dan kumpulkan objek yang terdeteksi
            frame, detected_objects = display_results(frame, results[0])
            
            # ini untuk Tampilkan video
            st_frame.image(frame, channels="RGB", use_column_width=True)
            
            # ini untuk menampilkan informasi deteksi
            if detected_objects:
                object_counts = Counter(detected_objects)
                detection_info = "\n".join([f"{obj}: {count}" for obj, count in object_counts.items()])
            else:
                detection_info = "No objects detected."

            st_detection_info.text(detection_info)  # Update detection info text

            # Keluar dari loop jika mode gaming dinonaktifkan
            if not st.session_state.detection_active:
                break
        
        cap.release()
    else:
        st.sidebar.warning("MATI CUYY: 🔴 Inactive")

if __name__ == "__main__":
    main()
