import streamlit as st
import cv2
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    model_path = "best.pt"
    model = YOLO(model_path)
    return model

def process_image(model, image_path):
    results = model(image_path)
    annotated_image = results[0].plot()

    bounding_boxes = []
    labels = []

    if results[0].boxes is not None:
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            bounding_boxes.append(box.cpu().numpy())
            labels.append(model.names[int(cls)])

    return annotated_image, bounding_boxes, labels

def draw_boxes(image_path, bounding_boxes, labels):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box, label in zip(bounding_boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red")

    return image

def analyze_mri(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    _, segmented = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    return edges, segmented

st.set_page_config(
    page_title="Alzheimer Detection", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

st.title("Alzheimer Detection Application")
st.sidebar.header("Tentang Aplikasi")
st.sidebar.write(
    "Aplikasi ini menggunakan YOLO untuk mendeteksi tanda-tanda Alzheimer pada gambar MRI."
    " \nSelain itu, aplikasi memberikan informasi tambahan tentang pengolahan citra."
)

st.write("## Mengenal Alzheimer")
st.write(
    "Alzheimer adalah penyakit progresif yang memengaruhi otak, menyebabkan penurunan daya ingat, kemampuan berpikir, dan perubahan perilaku. Berikut adalah kategori umum yang terdeteksi:"
)
st.markdown(
    "- Alzheimer Disease: Tahap lanjut dengan gejala yang jelas seperti kehilangan ingatan yang parah."
    "\n- Mild Cognitive Impairment: Gangguan kognitif ringan yang dapat menjadi awal dari Alzheimer."
    "\n- Cognitive Normal: Tidak ada tanda-tanda yang menunjukkan gangguan."
)

st.write("### Penyebab Alzheimer")
st.markdown(
    "- Penumpukan Protein Abnormal: Penumpukan beta-amyloid dan tau yang mengganggu fungsi sel otak."
    "\n- Faktor Genetik: Gen seperti APOE-e4 meningkatkan risiko."
    "\n- Usia Lanjut: Risiko meningkat setelah usia 65 tahun."
    "\n- Faktor Lingkungan dan Gaya Hidup: Kurang aktivitas fisik, pola makan buruk, dan kurang stimulasi mental."
)

st.write("### Tindakan Sesuai Diagnosis")
st.markdown(
    "- Alzheimer Disease: Konsultasikan dengan dokter spesialis, gunakan terapi obat seperti acetylcholinesterase inhibitors, dan berikan dukungan keluarga."
    "\n- Mild Cognitive Impairment: Perubahan gaya hidup sehat seperti olahraga teratur dan latihan otak."
    "\n- Cognitive Normal: Lanjutkan gaya hidup sehat dan lakukan pemeriksaan rutin."
)


st.write("### Unggah Gambar MRI untuk Analisis")
uploaded_file = st.file_uploader("Unggah file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path, caption="Gambar Asli", use_column_width=True)

    try:
        model = load_model()

        annotated_image, bounding_boxes, labels = process_image(model, image_path)

        if bounding_boxes:
            boxed_image = draw_boxes(image_path, bounding_boxes, labels)

            st.image(boxed_image, caption="Gambar dengan Deteksi", use_column_width=True)
            st.image(annotated_image, caption="Hasil Deteksi Anotasi", use_column_width=True)

        edges, segmented = analyze_mri(image_path)

        st.write("### Hasil Analisis Tambahan")
        st.image(edges, caption="Deteksi Tepi (Canny)", use_column_width=True, channels="GRAY")
        st.image(segmented, caption="Segmentasi (Threshold)", use_column_width=True, channels="GRAY")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

    os.remove(image_path)

st.sidebar.write("Dibuat oleh kelompok 3 Pengolahan Citra Grafika Komputer")
