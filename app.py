import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Pengaturan visualisasi
sns.set_style('darkgrid')

# ====== LOAD MODEL ======
model = load_model("model_combined.h5")

# ====== KELAS ======
class_names = {
    0: "all_benign",
    1: "all_early",
    2: "all_pre",
    3: "all_pro",
    4: "malaria_parasitized",
    5: "malaria_uninfected"
}

# ====== PREPROSES GAMBAR ======
def preprocess_image(image, size=(224, 224)):
    if image is None:
        return None
    image = image.resize(size)
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Jika RGBA
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

# ====== PREDIKSI ======
def predict_disease(name, age, gender, image):
    try:
        if image is None:
            return "❌ Silakan upload gambar sel terlebih dahulu.", None

        # Preprocessing
        img_processed = preprocess_image(image)
        prediction = model.predict(img_processed)[0]
        max_confidence = np.max(prediction)

        if max_confidence < 0.5:
            return "❌ Gambar yang diinput bukanlah gambar cell. Silakan upload gambar sel ALL atau Malaria yang valid.", None

        label_idx = np.argmax(prediction)
        class_name = class_names.get(label_idx, "Unknown")
        confidence = round(max_confidence * 100, 2)

        if class_name.startswith("all"):
            disease_type = "ALL (Acute Lymphoblastic Leukemia)"
            subtype = class_name.replace("all_", "").capitalize()
        elif class_name.startswith("malaria"):
            disease_type = "Malaria"
            subtype = class_name.replace("malaria_", "").capitalize()
        else:
            disease_type = "Unknown"
            subtype = class_name

        # Buat teks hasil
        result_md = (
            f"## 🔬 Hasil Prediksi Penyakit Sel\n\n"
            f"**👤 Nama:** {name}  \n"
            f"**🎂 Usia:** {age}  \n"
            f"**🚻 Jenis Kelamin:** {gender}  \n\n"
            f"**🧪 Penyakit:** {disease_type}  \n"
            f"**🔍 Subtype:** {subtype}  \n"
            f"**📊 Tingkat Keyakinan:** {confidence}%  \n"
        )

        # Plot probabilitas kelas
        probs = prediction * 100
        class_labels = [class_names[i].replace("all_", "ALL-").replace("malaria_", "Malaria-") for i in range(len(class_names))]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(class_labels, probs, color='teal', alpha=0.7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probabilitas (%)")
        ax.set_title("Probabilitas Kelas")
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.close(fig)

        return result_md, fig

    except Exception as e:
        return f"❌ Terjadi kesalahan saat memproses gambar.\nError: {str(e)}", None

# ====== GRADIO UI ======
with gr.Blocks(title="Klasifikasi Kombinasi ALL & Malaria") as demo:
    gr.Markdown("<h1 style='text-align: center; color: orange;'>🧬 Pendeteksi Penyakit Acute Lymphoblastic Leukemia (ALL) & Malaria 🧬</h1>")
    gr.Markdown("<h3 style='text-align: center;'>Upload gambar sel untuk prediksi otomatis penyakit ALL atau Malaria</h3>")

    with gr.Row():
        with gr.Column(scale=1):
            name_input = gr.Textbox(label="Nama Pasien")
            age_input = gr.Number(label="Usia Pasien", precision=0)
            gender_input = gr.Radio(["Laki-laki", "Perempuan"], label="Jenis Kelamin")
            image_input = gr.Image(type="pil", label="Upload Gambar Sel")
            btn = gr.Button("Prediksi", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Hasil Prediksi")
            output_plot = gr.Plot(label="Probabilitas Kelas")

    btn.click(
        fn=predict_disease,
        inputs=[name_input, age_input, gender_input, image_input],
        outputs=[output_text, output_plot]
    )

# ====== JALANKAN ======
if __name__ == "__main__":
    demo.launch()
