#!/usr/bin/env python3
import os, gi, ctypes, time, json, cv2, requests, numpy as np
gi.require_version('Gst', '1.0'); gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GObject, GstApp
Gst.init(None)

RTSP_URLS = os.getenv("RTSP_URLS", "rtsp://user:pass@192.168.1.10/stream1").split(",")
INSIGHT_API = os.getenv("INSIGHT_API", "http://insightface:18081/extract")
ATTEND_API  = os.getenv("ATTEND_API",  "http://api:8000")
SEND_EVERY_N = int(os.getenv("SEND_EVERY_N", "5"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))
CAMERA_IDS = [f"cam{i+1}" for i in range(len(RTSP_URLS))]  # đặt id camera

def build_pipeline():
    muxer = f"nvstreammux name=mux batch-size={len(RTSP_URLS)} width=1280 height=720 live-source=1"
    branches = []
    for i, uri in enumerate(RTSP_URLS):
        branches.append(f'uridecodebin uri="{uri}" live=true name=src{i} ')
        branches.append(f'src{i}.! queue ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! mux.sink_{i} ')
    pipeline = f'{" ".join(branches)} {muxer} ! nvvideoconvert ! video/x-raw,format=BGR ! appsink name=outsink emit-signals=true max-buffers=1 drop=true'
    return pipeline

pipeline = Gst.parse_launch(build_pipeline())
appsink = pipeline.get_by_name("outsink")
frame_count = 0

def post_extract(img_bgr):
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok: return None
    payload = {"img": enc.tobytes().hex(), "api_ver":"2"}
    r = requests.post(INSIGHT_API, json=payload, timeout=2.5)
    r.raise_for_status()
    return r.json()

def send_attendance(matches, camera_id):
    # matches: [{"employee_id":"E001","score":0.31,"bbox":[x1,y1,x2,y2]}...]
    payload = {"camera_id": camera_id, "matches": matches, "ts": int(time.time())}
    try:
        requests.post(f"{ATTEND_API}/log", json=payload, timeout=2.0)
    except Exception as e:
        print("[warn] attendance log failed:", e)

def on_new_sample(sink):
    global frame_count
    sample = sink.emit("pull-sample")
    if sample is None: return Gst.FlowReturn.OK
    buf = sample.get_buffer()
    caps = sample.get_caps()
    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success: return Gst.FlowReturn.OK

    try:
        s = caps.get_structure(0)
        w = s.get_value('width'); h = s.get_value('height')
        arr = memoryview(mapinfo.data)
        frame = (ctypes.c_ubyte * len(arr)).from_buffer_copy(arr)
        img = np.frombuffer(frame, dtype=np.uint8).reshape((h, w, 3))

        frame_count += 1
        if frame_count % SEND_EVERY_N != 0: 
            return Gst.FlowReturn.OK

        # Gọi InsightFace → lấy embeddings & bbox
        data = post_extract(img)
        if not data: return Gst.FlowReturn.OK
        faces = data.get("faces", [])
        if not faces: return Gst.FlowReturn.OK

        # gọi Attendance API để so khớp + log
        try:
            resp = requests.post(f"{ATTEND_API}/match", json={"faces": faces}, timeout=2.0)
            if resp.ok:
                matches = resp.json().get("matches", [])
                if matches:
                    send_attendance(matches, camera_id=CAMERA_IDS[0])
        except Exception as e:
            print("[warn] match API failed:", e)

    finally:
        buf.unmap(mapinfo)
    return Gst.FlowReturn.OK

appsink.connect("new-sample", on_new_sample)
pipeline.set_state(Gst.State.PLAYING)
bus = pipeline.get_bus()
print("[DeepStream] running…")
while True:
    msg = bus.timed_pop_filtered(1000, Gst.MessageType.ERROR | Gst.MessageType.EOS)
    if msg: 
        print("[BUS]", msg.type)
        break
