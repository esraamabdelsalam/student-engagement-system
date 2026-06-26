from dataclasses import dataclass, field
from typing import Any, List, Optional
import time

SEQUENCE_LENGTH = 16
BUFFER_TIMEOUT = 5 * 60  # 5 minutes


@dataclass
class StudentState:
    track_id: int

    # =========================
    # Identity (NO BUFFER)
    # =========================
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    embedding: Optional[Any] = None

    svc_confidence: float = 0.0
    cosine_similarity: float = 0.0
    recognition_confidence: float = 0.0

    # =========================
    # Emotion STREAM BUFFER
    # =========================
    emotion_buffer: List[str] = field(default_factory=list)
    current_emotion: Optional[str] = None
    emotion_confidence: float = 0.0

    # =========================
    # Engagement SEQUENCE BUFFER
    # =========================
    engagement_buffer: List[Any] = field(default_factory=list)
    current_engagement: Optional[str] = None
    engagement_confidence: float = 0.0

    # =========================
    # TIME
    # =========================
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # =========================
    # LIFECYCLE
    # =========================
    def update_last_seen(self):
        self.last_seen = time.time()

    def is_expired(self):
        return (time.time() - self.last_seen) > BUFFER_TIMEOUT

    # =========================
    # EMOTION STREAM
    # =========================
    def add_emotion(self, emotion: str):
        self.emotion_buffer.append(emotion)

    def clear_emotion_buffer(self):
        self.emotion_buffer.clear()

    def set_emotion_result(self, emotion: str, confidence: float = 0.0):
        self.current_emotion = emotion
        self.emotion_confidence = confidence

    def get_emotion_mode(self) -> Optional[str]:
        if not self.emotion_buffer:
            return None
        return max(set(self.emotion_buffer), key=self.emotion_buffer.count)

    # =========================
    # ENGAGEMENT SEQUENCE
    # =========================
    def add_frame(self, frame: Any):
        if len(self.engagement_buffer) < SEQUENCE_LENGTH:
            self.engagement_buffer.append(frame)

    def is_engagement_ready(self) -> bool:
        return len(self.engagement_buffer) == SEQUENCE_LENGTH

    def get_engagement_sequence(self) -> Optional[List[Any]]:
        if len(self.engagement_buffer) != SEQUENCE_LENGTH:
            return None
        return list(self.engagement_buffer)

    def set_engagement_result(self, label: str, confidence: float):
        self.current_engagement = label
        self.engagement_confidence = confidence

    def clear_engagement_buffer(self):
        self.engagement_buffer.clear()

    # =========================
    # SAFE CYCLE BOUNDARY
    # =========================
    def reset_engagement_cycle(self):
        """
        ONLY for engagement cycle reset (16-frame completion)
        """
        self.clear_engagement_buffer()

    def reset_emotion_cycle(self):
        """
        ONLY for emotion aggregation reset (mode computed)
        """
        self.clear_emotion_buffer()