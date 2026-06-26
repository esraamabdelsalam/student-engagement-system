from typing import Dict, List
from .student_state import StudentState


class StudentManager:
    def __init__(self):
        # track_id -> StudentState
        self.students: Dict[int, StudentState] = {}

    # =========================
    # GET OR CREATE
    # =========================
    def get(self, track_id: int) -> StudentState:
        if track_id not in self.students:
            self.students[track_id] = StudentState(track_id)
        return self.students[track_id]

    # =========================
    # REMOVE ONE
    # =========================
    def remove(self, track_id: int) -> None:
        self.students.pop(track_id, None)

    # =========================
    # CLEANUP EXPIRED
    # =========================
    def cleanup(self) -> List[int]:
        expired_ids = [
            track_id
            for track_id, student in self.students.items()
            if student.is_expired()
        ]

        for track_id in expired_ids:
            self.students.pop(track_id, None)

        return expired_ids

    # =========================
    # GET ALL ACTIVE
    # =========================
    def all_students(self) -> List[StudentState]:
        return list(self.students.values())

    # =========================
    # EXISTS CHECK
    # =========================
    def exists(self, track_id: int) -> bool:
        return track_id in self.students

    # =========================
    # SIZE
    # =========================
    def size(self) -> int:
        return len(self.students)


# =====================================================
# SINGLETON INSTANCE (IMPORTANT FIX FOR YOUR IMPORT)
# =====================================================
student_manager = StudentManager()