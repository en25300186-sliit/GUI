from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Sequence

try:  # pragma: no cover - exercised when CuPy is installed
    import cupy as cp
except ImportError:  # pragma: no cover - fallback for environments without CuPy
    cp = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback for environments without NumPy
    np = None


class ObjectState(IntEnum):
    ACTIVE = 1
    HIDDEN = 0
    OUTOFSCREEN = -1


@dataclass
class Object:
    x: float
    y: float
    width: float
    height: float
    z: float = 0.0
    state: ObjectState = ObjectState.ACTIVE
    on_hover: Optional[Callable[["Object"], None]] = None
    on_click: Optional[Callable[["Object"], None]] = None
    tensor_index: Optional[int] = None


class ObjectGroup(Object):
    def __init__(self, *items: Object, subitems: Optional[Sequence[Object]] = None, **kwargs):
        super().__init__(**kwargs)
        self.subitems: List[Object] = []
        for subitem in items:
            self.add(subitem)
        for subitem in subitems or []:
            self.add(subitem)

    def add(self, subitem: Object) -> None:
        if not isinstance(subitem, Object):
            raise TypeError("ObjectGroup accepts Object or ObjectGroup instances")
        self.subitems.append(subitem)


class NeuralWorld:
    _ROW_X = 0
    _ROW_Y = 1
    _ROW_HALF_W = 2
    _ROW_HALF_H = 3
    _ROW_Z = 4
    _ROW_STATE = 5

    def __init__(self, use_cupy: bool = True) -> None:
        if use_cupy and cp is not None:
            self.xp = cp
            self.backend = "cupy"
            self.world_tensor = self.xp.zeros((0, 6), dtype=self.xp.float32)
        elif np is not None:
            self.xp = np
            self.backend = "numpy"
            self.world_tensor = self.xp.zeros((0, 6), dtype=self.xp.float32)
        else:
            self.xp = None
            self.backend = "python"
            self.world_tensor: List[List[float]] = []
        self._objects: List[Object] = []
        self._index_to_object: Dict[int, Object] = {}
        self._last_hover_index: Optional[int] = None

    def register(self, obj: Object) -> int:
        if isinstance(obj, ObjectGroup):
            group_index = self._register_single(obj)
            for subitem in obj.subitems:
                self.register(subitem)
            return group_index
        return self._register_single(obj)

    def _register_single(self, obj: Object) -> int:
        row = [
            obj.x,
            obj.y,
            obj.width / 2.0,
            obj.height / 2.0,
            obj.z,
            float(obj.state),
        ]
        if self.backend == "python":
            self.world_tensor.append(row)
            index = len(self.world_tensor) - 1
        else:
            row_tensor = self.xp.asarray(row, dtype=self.xp.float32)
            self.world_tensor = self.xp.concatenate((self.world_tensor, row_tensor[None, :]), axis=0)
            index = int(self.world_tensor.shape[0] - 1)
        obj.tensor_index = index
        self._objects.append(obj)
        self._index_to_object[index] = obj
        return index

    @staticmethod
    def screen_to_world(mouse_x_px: float, mouse_y_px: float, width_px: float, height_px: float) -> Sequence[float]:
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Viewport dimensions must be positive")
        aspect = width_px / height_px
        x_ndc = ((mouse_x_px / width_px) * 2.0 - 1.0) * aspect
        y_ndc = 1.0 - (mouse_y_px / height_px) * 2.0
        return x_ndc, y_ndc

    @staticmethod
    def _has_handler(obj: Object, event: str) -> bool:
        if event == "hover":
            return obj.on_hover is not None
        if event == "click":
            return obj.on_click is not None
        return False

    def _to_scalar(self, value: object) -> object:
        if self.backend in {"python", "numpy"}:
            return value
        return self.xp.asnumpy(value)

    def hit_mask(self, x_world: float, y_world: float, *, event: str = "hover"):
        if (self.backend == "python" and not self.world_tensor) or (
            self.backend != "python" and self.world_tensor.shape[0] == 0
        ):
            return [] if self.backend == "python" else self.xp.asarray([], dtype=bool)

        if self.backend == "python":
            mask = []
            for row, obj in zip(self.world_tensor, self._objects):
                callback_ok = self._has_handler(obj, event)
                is_hit = (
                    row[self._ROW_STATE] == float(ObjectState.ACTIVE)
                    and abs(row[self._ROW_X] - x_world) <= row[self._ROW_HALF_W]
                    and abs(row[self._ROW_Y] - y_world) <= row[self._ROW_HALF_H]
                    and callback_ok
                )
                mask.append(is_hit)
            return mask

        rows = self.world_tensor
        active_mask = rows[:, self._ROW_STATE] == float(ObjectState.ACTIVE)
        within_x = self.xp.abs(rows[:, self._ROW_X] - x_world) <= rows[:, self._ROW_HALF_W]
        within_y = self.xp.abs(rows[:, self._ROW_Y] - y_world) <= rows[:, self._ROW_HALF_H]
        callback_mask = self.xp.ones(rows.shape[0], dtype=bool)
        for index, obj in self._index_to_object.items():
            if not self._has_handler(obj, event):
                callback_mask[index] = False

        return active_mask & within_x & within_y & callback_mask

    def topmost_hit(self, x_world: float, y_world: float, *, event: str = "hover") -> Optional[int]:
        mask = self.hit_mask(x_world, y_world, event=event)
        if self.backend == "python":
            candidate_indices = [i for i, hit in enumerate(mask) if hit]
            if not candidate_indices:
                return None
            return max(candidate_indices, key=lambda idx: (self.world_tensor[idx][self._ROW_Z], idx))

        if int(self._to_scalar(mask.sum())) == 0:
            return None

        candidate_indices = self.xp.where(mask)[0]
        candidate_z = self.world_tensor[candidate_indices, self._ROW_Z]
        best_local_idx = int(self._to_scalar(self.xp.argmax(candidate_z)))
        return int(self._to_scalar(candidate_indices[best_local_idx]))

    def dispatch(self, event: str, mouse_x_px: float, mouse_y_px: float, width_px: float, height_px: float) -> Optional[Object]:
        x_world, y_world = self.screen_to_world(mouse_x_px, mouse_y_px, width_px, height_px)
        winner = self.topmost_hit(x_world, y_world, event=event)

        if event == "hover":
            self._last_hover_index = winner

        if winner is None:
            return None

        obj = self._index_to_object[winner]
        if event == "hover" and obj.on_hover is not None:
            obj.on_hover(obj)
        elif event == "click" and obj.on_click is not None:
            obj.on_click(obj)
        return obj

    def win_proc(self, message: str, mouse_x_px: float, mouse_y_px: float, width_px: float, height_px: float) -> Optional[Object]:
        if message == "mouse_move":
            return self.dispatch("hover", mouse_x_px, mouse_y_px, width_px, height_px)
        if message == "mouse_click":
            return self.dispatch("click", mouse_x_px, mouse_y_px, width_px, height_px)
        return None
