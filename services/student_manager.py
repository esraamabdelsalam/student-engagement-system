from typing import Dict, List
from .student_state import StudentState


class StudentManager:
    def __init__(self):
        self.students: Dict[int, StudentState] = {}

    def get(self, track_id: int) -> StudentState:
        if track_id not in self.students:
            self.students[track_id] = StudentState(track_id)
        return self.students[track_id]

    def remove(self, track_id: int) -> None:
        self.students.pop(track_id, None)

    def cleanup(self) -> List[int]:
        expired = [
            tid for tid, s in self.students.items()
            if s.is_expired()
        ]
        for tid in expired:
            self.students.pop(tid, None)
        return expired

    def all_students(self):
        return list(self.students.values())


student_manager = StudentManager()