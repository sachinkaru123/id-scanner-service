"""FastAPI entrypoint: internal microservice for ID card OCR extraction."""
import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.parsers import parse_id_card
from app.pipeline import process_image
from app.quality import assess_quality
from app.roi_extract import extract_fields_via_roi

app = FastAPI(title="ID Scanner Service", version="0.1.0")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/scan")
async def scan_id(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    start = time.perf_counter()
    text, preview, ocr_data, cropped = process_image(image)
    fields = parse_id_card(text)

    # Targeted re-OCR of just the region next to each label. Only overrides
    # the regex-parsed value when it actually found something — the
    # whole-card parse above remains the fallback.
    roi_fields = extract_fields_via_roi(preview, ocr_data)
    for key, value in roi_fields.items():
        if value:
            fields[key] = value

    quality = assess_quality(cropped)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    ok, preview_bytes = cv2.imencode(".png", preview)
    preview_b64 = base64.b64encode(preview_bytes.tobytes()).decode("ascii") if ok else None

    return JSONResponse(content={
        "fields": fields,
        "quality": quality,
        "processing_time_ms": elapsed_ms,
        "preview_image": f"data:image/png;base64,{preview_b64}" if preview_b64 else None,
    })
