from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_3x4(data):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = np.array(img)

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, "Wajah tidak terdeteksi"

    # Ambil wajah terbesar
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, fw, fh = faces[0]

    # Jika ada wajah kedua yang hampir sebesar wajah utama,
    # tandai untuk dicek.
    if len(faces) > 1:
        _, _, fw2, fh2 = faces[1]
        if fw2 * fh2 >= fw * fh * 0.60:
            return None, "Lebih dari satu wajah"

    # Area vertikal: beri ruang kepala di atas dan badan di bawah
    cx = x + fw // 2
    top = max(0, y - int(fh * 0.75))
    bottom = min(h, y + int(fh * 3.0))

    # Target rasio 3:4
    target_ratio = 3 / 4
    crop_h = bottom - top
    crop_w = int(crop_h * target_ratio)

    # Jika terlalu lebar, batasi berdasarkan lebar gambar
    if crop_w > w:
        crop_w = w
        crop_h = int(crop_w / target_ratio)

    left = cx - crop_w // 2
    right = left + crop_w

    if left < 0:
        left = 0
        right = crop_w

    if right > w:
        right = w
        left = w - crop_w

    # Sesuaikan tinggi bila keluar batas
    if crop_h > h:
        crop_h = h
        crop_w = int(crop_h * target_ratio)
        left = max(0, cx - crop_w // 2)
        right = min(w, left + crop_w)
        top = 0
        bottom = crop_h

    crop = img[top:bottom, left:right]

    if crop.size == 0:
        return None, "Crop gagal"

    result = Image.fromarray(crop).resize(
        (600, 800),
        Image.Resampling.LANCZOS
    )

    output = io.BytesIO()
    result.save(output, format="JPEG", quality=95)
    output.seek(0)

    return output, None


@app.route("/", methods=["GET"])
def home():
    return "TKA FOTO SERVER AKTIF"


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "File tidak ditemukan"}), 400

    file = request.files["file"]
    data = file.read()

    result, error = crop_3x4(data)

    if error:
        return jsonify({
            "ok": False,
            "status": "PERLU_CEK",
            "error": error
        }), 422

    return send_file(
        result,
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="hasil_3x4.jpg"
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
