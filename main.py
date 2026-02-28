from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import subprocess
import shutil
import uuid
import os
import json

app = FastAPI()

UPLOAD_DIR = "api_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def run_script(command_list):
    result = subprocess.run(
        command_list,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise Exception(result.stderr)

    return result


# Senaryo 1
@app.post("/scenario1")
async def scenario1(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, file_id + ".jpg")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        run_script(["python", "main_puan.py", file_path])

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        result_path = f"output(puan)/{base_name}_scores.json"

        if not os.path.exists(result_path):
            return JSONResponse({"error": "Sonuç oluşturulamadı"}, status_code=500)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Senaryo 2
@app.post("/scenario2")
async def scenario2(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, file_id + ".jpg")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        run_script(["python", "main_v3.py", file_path])

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        result_path = f"output(duzenlenmis)/{base_name}_processed.json"

        if not os.path.exists(result_path):
            return JSONResponse({"error": "Sonuç oluşturulamadı"}, status_code=500)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Senaryo 3
@app.post("/scenario3")
async def scenario3(
    ocr_file: UploadFile = File(...),
    correct_file: UploadFile = File(...)
):
    try:
        file_id = str(uuid.uuid4())

        ocr_path = os.path.join(UPLOAD_DIR, f"{file_id}_ocr.json")
        correct_path = os.path.join(UPLOAD_DIR, f"{file_id}_correct.json")

        with open(ocr_path, "wb") as f:
            shutil.copyfileobj(ocr_file.file, f)

        with open(correct_path, "wb") as f:
            shutil.copyfileobj(correct_file.file, f)

        run_script([
            "python",
            "main_evaluate.py",
            ocr_path,
            correct_path
        ])

        os.makedirs("output_llm", exist_ok=True)

        result_file = None
        for filename in os.listdir("output_llm"):
            if file_id in filename:
                result_file = os.path.join("output_llm", filename)
                break

        if not result_file or not os.path.exists(result_file):
            return JSONResponse({"error": "LLM sonucu oluşmadı"}, status_code=500)

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Kontrol
@app.get("/health")
def health():
    return {"status": "ok"}