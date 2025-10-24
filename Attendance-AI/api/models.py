from sqlalchemy import Column, Integer, String, Float, LargeBinary, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base
from datetime import datetime
Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    employee_id = Column(String, unique=True, index=True)  # mã NV
    name = Column(String)

class FaceTemplate(Base):
    __tablename__ = "face_templates"
    id = Column(Integer, primary_key=True)
    employee_id = Column(String, index=True)
    dim = Column(Integer)
    embedding = Column(LargeBinary)  # float32 bytes
    __table_args__ = (UniqueConstraint('employee_id','id', name='uq_emp_template'),)

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    id = Column(Integer, primary_key=True)
    employee_id = Column(String, index=True)
    camera_id = Column(String)
    score = Column(Float)
    ts = Column(DateTime, default=datetime.utcnow)
