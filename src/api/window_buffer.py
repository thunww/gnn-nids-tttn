"""Gom flow (dong conn.log da doc, chua qua convert) thanh cua so truot cho suy luan real-time.

Mo phong dung logic cua graph.windowing.sliding_windows() (Giai doan 2, offline) nhung o dang
streaming: moi lan co 1 flow moi (Phase F tail conn.log), goi add() 1 lan; ham tra ve danh sach
window_size flow moi nhat khi da du de tao 1 cua so, nguoc lai tra ve None.
"""

from __future__ import annotations

from collections import deque


class SlidingWindowBuffer:
    def __init__(self, window_size: int, overlap: float = 0.5):
        self.window_size = window_size
        self.step = max(1, int(window_size * (1 - overlap)))
        self._buffer: deque[dict] = deque(maxlen=window_size)
        self._count = 0

    def add(self, row: dict) -> list[dict] | None:
        """Them 1 flow moi. Tra ve danh sach window_size flow gan nhat (1 cua so hoan chinh)
        moi khi du dieu kien truot (dung step = window_size * (1 - overlap)), nguoc lai None."""
        self._buffer.append(row)
        self._count += 1
        if self._count >= self.window_size and (self._count - self.window_size) % self.step == 0:
            return list(self._buffer)
        return None

    def flows_until_next_window(self) -> int:
        """So flow con thieu de co cua so tiep theo -- dung hien thi UI kieu 'dang cho x/N flow'."""
        if self._count < self.window_size:
            return self.window_size - self._count
        remainder = (self._count - self.window_size) % self.step
        return self.step - remainder if remainder else self.step
