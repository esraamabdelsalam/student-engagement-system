from dataclasses import dataclass, field
from typing import Any, List, Optional
import time
from collections import Counter

SEQUENCE_LENGTH = 16
BUFFER_TIMEOUT = 4.5 * 60  # 4.5 minutes


@dataclass
class StudentState:
    track_id: int

    # identity
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    recognition_confidence: float = 0.0

    # emotion buffer
    emotion_buffer: List[str] = field(default_factory=list)

    # engagement buffer
    engagement_buffer: List[Any] = field(default_factory=list)

    # time tracking
    last_seen: float = field(default_factory=time.time)

    # =========================
    # TIMEOUT RESET LOGIC
    # =========================
    def update_last_seen(self):
        now = time.time()

        if now - self.last_seen > BUFFER_TIMEOUT:
            self.reset_emotion_cycle()
            self.reset_engagement_cycle()

        self.last_seen = now

    def is_expired(self):
        return (time.time() - self.last_seen) > BUFFER_TIMEOUT

    # =========================
    # EMOTION BUFFER
    # =========================
    def add_emotion(self, emotion: str):
        # اجمع حتى 16 فقط
        if len(self.emotion_buffer) < SEQUENCE_LENGTH:
            self.emotion_buffer.append(emotion)

    def get_emotion_mode(self) -> Optional[str]:
        if len(self.emotion_buffer) < SEQUENCE_LENGTH:
            return None

        return Counter(self.emotion_buffer).most_common(1)[0][0]

    def get_emotion_result(self) -> Optional[str]:
        result = self.get_emotion_mode()

        if result is None:
            return None

        self.reset_emotion_cycle()
        return result

    def reset_emotion_cycle(self):
        self.emotion_buffer.clear()

    # =========================
    # ENGAGEMENT BUFFER
    # =========================
    def add_frame(self, frame: Any):
        # اجمع حتى 16 فقط
        if len(self.engagement_buffer) < SEQUENCE_LENGTH:
            self.engagement_buffer.append(frame)

    def get_engagement_sequence(self) -> Optional[List[Any]]:
        if len(self.engagement_buffer) < SEQUENCE_LENGTH:
            return None

        seq = self.engagement_buffer.copy()

        # بعد الإرسال للموديل ابدأ دورة جديدة
        self.reset_engagement_cycle()

        return seq

    def reset_engagement_cycle(self):
        self.engagement_buffer.clear()