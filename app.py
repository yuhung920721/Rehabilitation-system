from __future__ import annotations

from flask import Flask, Response, jsonify, render_template, request

from rehab_app.camera import CameraService


app = Flask(__name__)
camera_service = CameraService()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(
        camera_service.frame_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def api_status():
    return jsonify(camera_service.get_status())


@app.post("/api/reset")
def api_reset():
    camera_service.reset()
    return jsonify(camera_service.get_status())


@app.post("/api/target")
def api_target():
    data = request.get_json(silent=True) or {}
    target_count = int(data.get("target_count", 5))
    camera_service.set_target_count(target_count)
    return jsonify(camera_service.get_status())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
