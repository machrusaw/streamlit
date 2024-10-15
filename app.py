import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
import io
import os
import gdown

# Fungsi untuk memuat model generator GAN yang disimpan dalam Google Drive
@st.cache_resource
def load_gan_model_from_drive(model_url, output_path):
    if not os.path.exists(output_path):
        gdown.download(model_url, output_path, quiet=False)
    model = tf.keras.models.load_model(output_path)
    return model

# Fungsi untuk mengambil model GAN berdasarkan pilihan dropdown
def get_gan_model(model_option):
    if model_option == "Model Epoch 40":
        model_url = 'https://drive.google.com/uc?id=122vKlan3zBfSHA4mq4OJiPQpSM9CqDDa'  # Ganti dengan ID file model Epoch 40
        model_path = 'generator_epoch40.h5'
    elif model_option == "Model Epoch 100":
        model_url = 'https://drive.google.com/uc?id=1q4LNa__tLMV9C_NDH2MekP2XVaLKNXT3'  # Ganti dengan ID file model Epoch 100
        model_path = 'generator_epoch100.h5'


    # Load model dari Google Drive
    generator = load_gan_model_from_drive(model_url, model_path)
    return generator

# Fungsi untuk melakukan prediksi pada gambar grayscale dengan model GAN
def predict_image(generator, grayscale_image):
    grayscale_image = np.array(grayscale_image)
    grayscale_image = cv.resize(grayscale_image, (256, 256))
    grayscale_image = grayscale_image.astype('float32') / 255.0
    grayscale_image = np.expand_dims(grayscale_image, axis=-1)  # Ubah jadi 1 channel
    grayscale_image = np.repeat(grayscale_image, 3, axis=-1)  # Ubah jadi 3 channel
    grayscale_image = np.expand_dims(grayscale_image, axis=0)  # Tambah dimensi batch
    prediction = generator.predict(grayscale_image)
    prediction = np.clip(prediction[0], 0, 1)
    return prediction

# Fungsi untuk memuat model CNN Pretrained Caffe
def load_caffe_model():
    DIR = 'model'
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    PROTOTXT_PATH = os.path.join(DIR, 'colorization_deploy_v2.prototxt')
    POINTS_PATH = os.path.join(DIR, 'pts_in_hull.npy')
    MODEL_PATH = os.path.join(DIR, 'colorization_release_v2.caffemodel')

    if not (os.path.exists(PROTOTXT_PATH) and os.path.exists(POINTS_PATH) and os.path.exists(MODEL_PATH)):
        st.write("Downloading model files...")
        PROTOTXT_URL = 'https://drive.google.com/uc?id=1DZ4cFBYC3_KjOn2ayrhnk2XKHt6E54EJ'
        POINTS_URL = 'https://drive.google.com/uc?id=1Qh54l1Jhh5psiytgsv9WmJVByjpHdF8o'
        MODEL_URL = 'https://drive.google.com/uc?id=1RCb6SJN2T5tdrpPUXEx0L4GBaTtc2OcL'

        gdown.download(PROTOTXT_URL, PROTOTXT_PATH, quiet=False)
        gdown.download(POINTS_URL, POINTS_PATH, quiet=False)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    net = cv.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    pts = np.load(POINTS_PATH)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    return net

# Fungsi untuk prediksi gambar menggunakan model CNN Pretrained Caffe
def colorize_image(net, image_resized):
    scaled = image_resized.astype("float32") / 255.0
    lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)
    resized = cv.resize(lab, (224, 224))  # Resize for model input
    L = cv.split(resized)[0]
    L -= 50

    net.setInput(cv.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv.resize(ab, (image_resized.shape[1], image_resized.shape[0]))

    L = cv.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

# Streamlit layout
st.set_page_config(page_title="GAN & CNN Image Colorization", layout="wide")

# Dropdown untuk memilih metode
method = st.sidebar.selectbox(
    "Pilih metode",
    ["GAN", "CNN Pretrained Caffe"]
)

# Jika metode yang dipilih adalah GAN
if method == "GAN":
    # Sidebar untuk unggah gambar
    uploaded_files = st.sidebar.file_uploader("Pilih gambar", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Jika tidak ada gambar yang diunggah, tampilkan teks sambutan
    if not uploaded_files:
        st.title("Image Colorization with GAN (Pix2Pix)")
        st.markdown("""
            ### Selamat datang di aplikasi Pewarnaan Gambar Menggunakan GAN
            Unggah satu atau beberapa gambar grayscale Anda di sidebar, dan model kami akan menghasilkan gambar berwarna untuk Anda.
            Gunakan tombol di bawah untuk mengunduh gambar hasil prediksi.
        """)
        
    else:
        # Jika gambar telah diunggah, hilangkan teks sambutan
        st.title("Hasil Prediksi Gambar Berwarna menggunakan GAN (Pix2Pix)")

    # Dropdown untuk memilih model GAN
    model_option = st.sidebar.selectbox(
        "Pilih model GAN",
        ["Model Epoch 40", "Model Epoch 100"]
    )

    # Load model GAN yang diambil dari Google Drive
    generator = get_gan_model(model_option)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            grayscale_image = Image.open(uploaded_file).convert("L")

            # Prediksi gambar berwarna dengan GAN
            prediction_image = predict_image(generator, grayscale_image)

            # Ubah ukuran gambar agar konsisten
            target_size = (256, 256)
            grayscale_image = grayscale_image.resize(target_size)
            input_image = Image.open(uploaded_file).resize(target_size)
            prediction_image = Image.fromarray((prediction_image * 255).astype(np.uint8)).resize(target_size)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(input_image, caption="Gambar Inputan", use_column_width=True)
            with col2:
                st.image(grayscale_image, caption="Gambar Grayscale", use_column_width=True)
            with col3:
                st.image(prediction_image, caption="Gambar Hasil Prediksi", use_column_width=True)

            save_path = f"predicted_{uploaded_file.name}"
            img_bytes = io.BytesIO()
            prediction_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            st.download_button(
                label=f"Download Gambar Hasil Prediksi",
                data=img_bytes,
                file_name=save_path,
                mime="image/png"
            )

# Jika metode yang dipilih adalah CNN Pretrained Caffe
elif method == "CNN Pretrained Caffe":
    # Sidebar untuk unggah gambar
    uploaded_files = st.sidebar.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    # Jika tidak ada gambar yang diunggah, tampilkan teks sambutan
    if not uploaded_files:
        st.title("Image Colorization with CNN Pretrained Caffe")
        st.markdown("""
    ### Selamat datang di aplikasi Pewarnaan Gambar Menggunakan CNN Pretrained Caffe
    Unggah satu atau beberapa gambar di sidebar, dan model CNN Pretrained Caffe kami akan melakukan pewarnaan otomatis pada gambar grayscale Anda. 
    Metode ini menggunakan jaringan syaraf tiruan yang sudah dilatih sebelumnya untuk mengubah gambar hitam putih menjadi berwarna. 
    Setelah pewarnaan selesai, Anda dapat mengunduh gambar hasil prediksi menggunakan tombol di bawah.
    """)
    else:
        # Jika gambar telah diunggah, hilangkan teks sambutan
        st.title("Hasil Prediksi Gambar Berwarna menggunakan CNN Pretrained Caffe")
    # Load model CNN Pretrained Caffe
    net = load_caffe_model()

    

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
            image_resized = cv.resize(image, (256, 256))
            gray_image = cv.cvtColor(image_resized, cv.COLOR_BGR2GRAY)

            # Colorize the image using CNN Pretrained Caffe
            colorized_image = colorize_image(net, image_resized)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image_resized, channels="BGR", caption="Gambar Inputan", use_column_width=True)
            with col2:
                st.image(gray_image, channels="GRAY", caption="Gambar Grayscale", use_column_width=True)
            with col3:
                st.image(colorized_image, channels="BGR", caption="Gambar Hasil Prediksi", use_column_width=True)

            # Option to download the colorized image
            result_image = cv.imencode('.png', colorized_image)[1].tobytes()
            st.download_button(
                label=f"Download Colorized Image - {uploaded_file.name}",
                data=result_image,
                file_name=f"colorized_{uploaded_file.name}",
                mime="image/png"
            )