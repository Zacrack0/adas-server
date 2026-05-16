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
    "person":        {"alerta": "¡Cuidado! Peatón detectado",         "nivel": "critico"},
    "bicycle":       {"alerta": "¡Atención! Ciclista en la vía",      "nivel": "critico"},
    "motorcycle":    {"alerta": "¡Cuidado! Motociclista detectado",   "nivel": "critico"},
    "dog":           {"alerta": "¡Precaución! Animal en la vía",      "nivel": "critico"},
    "cat":           {"alerta": "¡Precaución! Animal en la vía",      "nivel": "critico"},
    "car":           {"alerta": "Vehículo detectado al frente",       "nivel": "advertencia"},
    "truck":         {"alerta": "Camión detectado, mantén distancia", "nivel": "advertencia"},
    "bus":           {"alerta": "Autobús detectado en la vía",        "nivel": "advertencia"},
    "stop sign":     {"alerta": "Señal de alto detectada",            "nivel": "advertencia"},
    "traffic light": {"alerta": "Semáforo detectado",                 "nivel": "advertencia"},
}

# Almacena la última distancia recibida del ESP32
ultima_distancia = {"valor": None}

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ADAS Server activo"})

# ── Endpoint para el ESP32 ──────────────────────────────────────────
@app.route("/distancia", methods=["POST"])
def recibir_distancia():
    try:
        dist_str = request.form.get("distancia", None)
        if dist_str and dist_str.isdigit():
            ultima_distancia["valor"] = int(dist_str)
            print(f"ESP32 → {ultima_distancia['valor']} cm")
        alerta = None
        d = ultima_distancia["valor"]
        if d and d < 50:
            alerta = "¡Obstáculo muy cercano!"
        elif d and d < 150:
            alerta = "Obstáculo detectado cerca"
        return jsonify({"distancia": d, "alerta": alerta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Endpoint para que la app consulte la distancia actual ───────────
@app.route("/distancia_actual", methods=["GET"])
def distancia_actual():
    return jsonify({"distancia": ultima_distancia["valor"]})

# ── Endpoint principal: imagen + distancia → detección YOLO ─────────
@app.route("/detectar", methods=["POST"])
def detectar():
    print("\n===== /detectar =====")
    try:
        if "imagen" not in request.files:
            return jsonify({"error": "No se recibió imagen"}), 400

        file = request.files["imagen"]
        imagen = Image.open(file).convert("RGB")

        dist_str = request.form.get("distancia", None)
        distancia_cm = int(dist_str) if dist_str and dist_str.isdigit() else ultima_distancia["valor"]
        print(f"  Distancia usada: {distancia_cm} cm | Imagen: {imagen.size}")

        img_array = np.array(imagen)
        results = model(img_array, conf=0.35)
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

        # Alerta de distancia crítica aunque no haya objeto reconocido
        if distancia_cm and distancia_cm < 50:
            msg = f"¡Obstáculo muy cercano a {distancia_cm} centímetros!"
            if msg not in critical_texts:
                critical_texts.insert(0, msg)

        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        hay_critico = any(a["nivel"] == "critico" for a in alerts)

        print(f"  Objetos: {len(alerts)} | Críticos: {hay_critico}")

        return jsonify({
            "alertas": [a["alert"] for a in alerts],
            "detecciones": alerts,
            "hay_critico": hay_critico,
            "distancia": distancia_cm,
            "imagen_anotada": img_b64
        })

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
