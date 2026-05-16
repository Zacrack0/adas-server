import torch
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import base64
import io
import os
from gtts import gTTS
import tempfile
import traceback
import cv2

print("Cargando modelo YOLOv8...")
model = YOLO("yolov8n.pt")
print("Modelo cargado.")

ADAS_CLASES = {
    "person":     {"alerta": "¡Cuidado! Peatón detectado",        "nivel": "critico"},
    "bicycle":    {"alerta": "¡Atención! Ciclista en la vía",     "nivel": "critico"},
    "motorcycle": {"alerta": "¡Cuidado! Motociclista detectado",  "nivel": "critico"},
    "dog":        {"alerta": "¡Precaución! Animal en la vía",     "nivel": "critico"},
    "car":        {"alerta": "Vehículo detectado al frente",      "nivel": "advertencia"},
    "truck":      {"alerta": "Camión detectado, mantén distancia","nivel": "advertencia"},
    "bus":        {"alerta": "Autobús detectado en la vía",       "nivel": "advertencia"},
    "stop sign":  {"alerta": "Señal de alto detectada",           "nivel": "advertencia"},
    "traffic light": {"alerta": "Semáforo detectado",             "nivel": "advertencia"},
}

def detectar_con_distancia(imagen_pil, distancia_cm=None, conf=0.35):
    img_array = np.array(imagen_pil)
    results = model(img_array, conf=conf)
    result = results[0]

    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    alerts = []
    critical_texts = []

    for box in result.boxes:
        cls_name = result.names[int(box.cls)].lower()
        confidence = float(box.conf)

        if cls_name in ADAS_CLASES:
            info = ADAS_CLASES[cls_name]
            alerta = info["alerta"]

            if distancia_cm and distancia_cm < 300 and info["nivel"] == "critico":
                alerta += f" a {distancia_cm} centímetros"

            if info["nivel"] == "critico":
                critical_texts.append(alerta)

            alerts.append({
                "object": cls_name,
                "confidence": round(confidence, 2),
                "alert": alerta,
                "nivel": info["nivel"]
            })

    # Generar audio si hay alertas críticas
    audio_b64 = None
    if critical_texts:
        try:
            texto_audio = ". ".join(critical_texts)
            tts = gTTS(text=texto_audio, lang='es', slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                tts.save(fp.name)
                audio_path = fp.name
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(audio_path)
        except Exception as e:
            print(f"Error generando audio: {e}")

    return annotated_pil, alerts, audio_b64


app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ADAS Server corriendo", "version": "1.0"})

@app.route("/detectar", methods=["POST"])
def detectar():
    print("\n===== Petición /detectar =====")
    try:
        # Aceptar imagen desde archivo o base64
        imagen = None

        if "imagen" in request.files:
            file = request.files["imagen"]
            imagen = Image.open(file).convert("RGB")
            print(f"  Imagen recibida por archivo: {file.filename}")

        elif "imagen_b64" in request.form:
            img_data = base64.b64decode(request.form["imagen_b64"])
            imagen = Image.open(io.BytesIO(img_data)).convert("RGB")
            print("  Imagen recibida por base64")

        else:
            return jsonify({"error": "No se recibió imagen"}), 400

        distancia_str = request.form.get("distancia", None)
        distancia_cm = int(distancia_str) if distancia_str and distancia_str.isdigit() else None
        print(f"  Distancia: {distancia_cm} cm")

        annotated_image, alerts, audio_b64 = detectar_con_distancia(imagen, distancia_cm=distancia_cm)

        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        hay_critico = any(a["nivel"] == "critico" for a in alerts)

        print(f"  Objetos detectados: {len(alerts)} | Críticos: {hay_critico}")

        return jsonify({
            "alertas": [a["alert"] for a in alerts],
            "detecciones": alerts,
            "hay_critico": hay_critico,
            "distancia": distancia_cm,
            "imagen_anotada": img_b64,
            "audio_b64": audio_b64
        })

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Endpoint solo para distancia (desde ESP32 sin imagen)
@app.route("/distancia", methods=["POST"])
def solo_distancia():
    try:
        distancia_str = request.form.get("distancia", None)
        distancia_cm = int(distancia_str) if distancia_str else None
        print(f"Distancia recibida: {distancia_cm} cm")

        alerta = None
        if distancia_cm and distancia_cm < 50:
            alerta = "¡Obstáculo muy cercano!"
        elif distancia_cm and distancia_cm < 150:
            alerta = "Obstáculo detectado cerca"

        return jsonify({
            "distancia": distancia_cm,
            "alerta": alerta
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
