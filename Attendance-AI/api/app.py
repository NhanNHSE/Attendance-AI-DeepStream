import os, io, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Employee, FaceTemplate, AttendanceLog
from matching import load_gallery_from_db, match_faces

IFREST_URL = os.getenv("IFREST_URL","http://insightface:18081")
DB_URL = os.getenv("DB_URL","sqlite:///./data/attendance.db")
THR = float(os.getenv("MATCH_THRESHOLD","0.42"))
MIN_FACE = int(os.getenv("MIN_FACE","112"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC","300"))

engine = create_engine(DB_URL, future=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

app = FastAPI(title="Attendance API")

class EnrollReq(BaseModel):
    employee_id: str
    name: str | None = None
    embeddings: list[list[float]]  # có thể gửi nhiều embedding để tăng ổn định

class MatchReq(BaseModel):
    faces: list[dict]  # từ InsightFace-REST /extract (api_ver=2)

class LogReq(BaseModel):
    camera_id: str
    matches: list[dict]
    ts: int

@app.on_event("startup")
def _startup():
    with SessionLocal() as s:
        load_gallery_from_db(s, FaceTemplate)

@app.post("/enroll")
def enroll(req: EnrollReq):
    # Lưu employee + template
    with SessionLocal() as s:
        emp = s.query(Employee).filter_by(employee_id=req.employee_id).first()
        if not emp:
            emp = Employee(employee_id=req.employee_id, name=req.name or req.employee_id)
            s.add(emp); s.commit()
        for emb in req.embeddings:
            e = np.asarray(emb, dtype=np.float32)
            s.add(FaceTemplate(employee_id=req.employee_id, dim=e.shape[0], embedding=e.tobytes()))
        s.commit()
        load_gallery_from_db(s, FaceTemplate)
    return {"ok": True}

@app.post("/match")
def match(req: MatchReq):
    # Lọc mặt nhỏ + lấy embeddings
    embs = []
    bboxes = []
    for f in req.faces:
        bbox = f.get("bbox") or f.get("bbox_xyxy")
        if bbox and (bbox[2]-bbox[0] < MIN_FACE or bbox[3]-bbox[1] < MIN_FACE):
            continue
        emb = f.get("embedding")
        if emb: 
            embs.append(np.asarray(emb, dtype=np.float32))
            bboxes.append(bbox or [0,0,0,0])
    pair = match_faces(embs, thr=THR, cooldown_sec=COOLDOWN_SEC)
    # ghép lại theo thứ tự
    matches = []
    i = 0
    for p in pair:
        emp, dist = p
        matches.append({"employee_id": emp, "score": dist, "bbox": bboxes[i]})
        i += 1
    return {"matches": matches}

@app.post("/log")
def log_attendance(req: LogReq):
    with SessionLocal() as s:
        for m in req.matches:
            s.add(AttendanceLog(employee_id=m["employee_id"], camera_id=req.camera_id, score=m["score"]))
        s.commit()
    return {"ok": True, "count": len(req.matches)}
