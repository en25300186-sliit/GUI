from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised when CuPy is installed
    import cupy as cp
except ImportError:  # pragma: no cover - fallback for environments without CuPy
    cp = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback for environments without NumPy
    np = None

try:  # pragma: no cover - only exercised on Windows with pywin32 installed
    import win32api
    import win32con
    import win32gui
except ImportError:  # pragma: no cover - fallback for environments without pywin32
    win32api = None
    win32con = None
    win32gui = None


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

    def __init__(self, use_cupy: bool = True, initial_capacity: int = 10_000, growth_chunk: int = 10_000) -> None:
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if growth_chunk <= 0:
            raise ValueError("growth_chunk must be positive")
        self._growth_chunk = int(growth_chunk)
        self._capacity = int(initial_capacity)
        self._size = 0
        self._global_dirty = False
        if use_cupy and cp is not None:
            self.xp = cp
            self.backend = "cupy"
            self.world_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._global_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._parent_index = self.xp.full((self._capacity,), -1, dtype=self.xp.int32)
        elif np is not None:
            self.xp = np
            self.backend = "numpy"
            self.world_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._global_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._parent_index = self.xp.full((self._capacity,), -1, dtype=self.xp.int32)
        else:
            self.xp = None
            self.backend = "python"
            self.world_tensor: List[List[float]] = []
            self._python_global_rows: List[List[float]] = []
            self._python_parent_index: List[int] = []
        self._objects: List[Object] = []
        self._index_to_object: Dict[int, Object] = {}
        self._last_hover_index: Optional[int] = None

    @property
    def size(self) -> int:
        return self._size if self.backend != "python" else len(self.world_tensor)

    @property
    def capacity(self) -> int:
        return self._capacity if self.backend != "python" else len(self.world_tensor)

    def register(self, obj: Object, _parent_index: int = -1) -> int:
        if isinstance(obj, ObjectGroup):
            group_index = self._register_single(obj, parent_index=_parent_index)
            for subitem in obj.subitems:
                self.register(subitem, _parent_index=group_index)
            return group_index
        return self._register_single(obj, parent_index=_parent_index)

    def _ensure_capacity(self, minimum_capacity: int) -> None:
        if self.backend == "python":
            return
        if minimum_capacity <= self._capacity:
            return
        new_capacity = self._capacity
        while new_capacity < minimum_capacity:
            new_capacity += self._growth_chunk
        expand_by = new_capacity - self._capacity
        zeros = self.xp.zeros((expand_by, 6), dtype=self.xp.float32)
        self.world_tensor = self.xp.concatenate((self.world_tensor, zeros), axis=0)
        self._global_tensor = self.xp.concatenate((self._global_tensor, zeros), axis=0)
        parent_padding = self.xp.full((expand_by,), -1, dtype=self.xp.int32)
        self._parent_index = self.xp.concatenate((self._parent_index, parent_padding), axis=0)
        self._capacity = new_capacity

    def _register_single(self, obj: Object, parent_index: int = -1) -> int:
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
            self._python_parent_index.append(parent_index)
            self._python_global_rows.append(row[:])
            index = len(self.world_tensor) - 1
        else:
            index = self._size
            self._ensure_capacity(index + 1)
            row_tensor = self.xp.asarray(row, dtype=self.xp.float32)
            self.world_tensor[index, :] = row_tensor
            self._parent_index[index] = int(parent_index)
            self._global_tensor[index, :] = row_tensor
            self._size += 1
        obj.tensor_index = index
        self._objects.append(obj)
        self._index_to_object[index] = obj
        self._global_dirty = True
        return index

    def _sync_global_transforms(self) -> None:
        if not self._global_dirty:
            return

        if self.backend == "python":
            state = [0] * len(self.world_tensor)

            def _resolve(index: int) -> None:
                if state[index] == 2:
                    return
                if state[index] == 1:
                    raise ValueError("Hierarchy cycle detected")
                state[index] = 1
                local_row = self.world_tensor[index]
                parent_idx = self._python_parent_index[index]
                global_row = local_row[:]
                if parent_idx >= 0:
                    if parent_idx >= len(self.world_tensor):
                        raise ValueError("Parent index is out of bounds")
                    _resolve(parent_idx)
                    parent_global = self._python_global_rows[parent_idx]
                    global_row[self._ROW_X] = local_row[self._ROW_X] + parent_global[self._ROW_X]
                    global_row[self._ROW_Y] = local_row[self._ROW_Y] + parent_global[self._ROW_Y]
                    global_row[self._ROW_Z] = local_row[self._ROW_Z] + parent_global[self._ROW_Z]
                self._python_global_rows[index] = global_row
                state[index] = 2

            for idx in range(len(self.world_tensor)):
                _resolve(idx)
            self._global_dirty = False
            return

        if self._size == 0:
            self._global_dirty = False
            return
        active_local = self.world_tensor[: self._size]
        active_global = self._global_tensor[: self._size]
        active_global[:, :] = active_local
        state = [0] * self._size

        def _resolve(index: int) -> None:
            if state[index] == 2:
                return
            if state[index] == 1:
                raise ValueError("Hierarchy cycle detected")
            state[index] = 1
            parent_idx = int(self._to_scalar(self._parent_index[index]))
            if parent_idx >= 0:
                if parent_idx >= self._size:
                    raise ValueError("Parent index is out of bounds")
                _resolve(parent_idx)
                active_global[index, self._ROW_X] = active_local[index, self._ROW_X] + active_global[parent_idx, self._ROW_X]
                active_global[index, self._ROW_Y] = active_local[index, self._ROW_Y] + active_global[parent_idx, self._ROW_Y]
                active_global[index, self._ROW_Z] = active_local[index, self._ROW_Z] + active_global[parent_idx, self._ROW_Z]
            state[index] = 2

        for idx in range(self._size):
            _resolve(idx)
        self._global_dirty = False

    def global_row(self, index: int) -> Optional[Sequence[float]]:
        if index < 0 or index >= self.size:
            return None
        self._sync_global_transforms()
        if self.backend == "python":
            return self._python_global_rows[index][:]
        return self._to_scalar(self._global_tensor[index, :]).tolist()

    def iter_render_rows(self) -> Iterable[Tuple[Object, Sequence[float]]]:
        self._sync_global_transforms()
        if self.backend == "python":
            for obj, row in zip(self._objects, self._python_global_rows):
                yield obj, row
            return

        active_rows = self._global_tensor[: self._size]
        if self.backend == "cupy":
            active_rows_cpu = cp.asnumpy(active_rows)
        else:
            active_rows_cpu = active_rows
        for idx, obj in enumerate(self._objects):
            yield obj, active_rows_cpu[idx].tolist()

    @staticmethod
    def screen_to_world(mouse_x_px: float, mouse_y_px: float, width_px: float, height_px: float) -> Sequence[float]:
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Viewport dimensions must be positive")
        aspect = width_px / height_px
        x_world = ((mouse_x_px / width_px) * 2.0 - 1.0) * aspect
        y_world = 1.0 - (mouse_y_px / height_px) * 2.0
        return x_world, y_world

    @staticmethod
    def _has_handler(obj: Object, event: str) -> bool:
        if event == "hover":
            return obj.on_hover is not None
        if event == "click":
            return obj.on_click is not None
        return False

    def _to_scalar(self, value: Any) -> Any:
        if self.backend in {"python", "numpy"}:
            return value
        return self.xp.asnumpy(value)

    def hit_mask(self, x_world: float, y_world: float, *, event: str = "hover"):
        if (self.backend == "python" and not self.world_tensor) or (self.backend != "python" and self._size == 0):
            return [] if self.backend == "python" else self.xp.asarray([], dtype=bool)

        self._sync_global_transforms()

        if self.backend == "python":
            mask = []
            for row, obj in zip(self._python_global_rows, self._objects):
                callback_ok = self._has_handler(obj, event)
                is_hit = (
                    row[self._ROW_STATE] == float(ObjectState.ACTIVE)
                    and abs(row[self._ROW_X] - x_world) <= row[self._ROW_HALF_W]
                    and abs(row[self._ROW_Y] - y_world) <= row[self._ROW_HALF_H]
                    and callback_ok
                )
                mask.append(is_hit)
            return mask

        rows = self._global_tensor[: self._size]
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
            return max(candidate_indices, key=lambda idx: (self._python_global_rows[idx][self._ROW_Z], idx))

        if int(self._to_scalar(mask.sum())) == 0:
            return None

        candidate_indices = self.xp.where(mask)[0]
        candidate_z = self._global_tensor[candidate_indices, self._ROW_Z]
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


class Win32Renderer:
    def __init__(
        self,
        world: NeuralWorld,
        *,
        title: str = "Neural UI",
        width: int = 1280,
        height: int = 720,
        background_color: int = 0x202020,
        object_color: int = 0x4080FF,
        hover_color: int = 0x40D880,
    ) -> None:
        if win32gui is None or win32con is None or win32api is None:
            raise RuntimeError("pywin32 is required to use Win32Renderer")
        self.world = world
        self.title = title
        self.width = width
        self.height = height
        self.background_color = background_color
        self.object_color = object_color
        self.hover_color = hover_color
        self._class_name = f"NeuralUIWindowClass_{id(self)}"
        self._wnd_proc = self._build_wnd_proc()
        self.hwnd: Optional[int] = None

    @staticmethod
    def world_to_screen(
        x_world: float, y_world: float, width_px: int, height_px: int, half_w_world: float, half_h_world: float
    ) -> Tuple[int, int, int, int]:
        aspect = width_px / height_px
        left_ndc = (x_world - half_w_world) / aspect
        right_ndc = (x_world + half_w_world) / aspect
        top_ndc = y_world + half_h_world
        bottom_ndc = y_world - half_h_world
        left = int((left_ndc + 1.0) * 0.5 * width_px)
        right = int((right_ndc + 1.0) * 0.5 * width_px)
        top = int((1.0 - (top_ndc + 1.0) * 0.5) * height_px)
        bottom = int((1.0 - (bottom_ndc + 1.0) * 0.5) * height_px)
        return left, top, right, bottom

    def _draw_world(self, hdc: int) -> None:
        rect = win32gui.GetClientRect(self.hwnd)
        bg_brush = win32gui.CreateSolidBrush(self.background_color)
        win32gui.FillRect(hdc, rect, bg_brush)
        win32gui.DeleteObject(bg_brush)

        for obj, row in self.world.iter_render_rows():
            if row[NeuralWorld._ROW_STATE] != float(ObjectState.ACTIVE):
                continue
            color = self.hover_color if obj.tensor_index == self.world._last_hover_index else self.object_color
            brush = win32gui.CreateSolidBrush(color)
            old_brush = win32gui.SelectObject(hdc, brush)
            left, top, right, bottom = self.world_to_screen(
                row[NeuralWorld._ROW_X],
                row[NeuralWorld._ROW_Y],
                self.width,
                self.height,
                row[NeuralWorld._ROW_HALF_W],
                row[NeuralWorld._ROW_HALF_H],
            )
            win32gui.Rectangle(hdc, left, top, right, bottom)
            win32gui.SelectObject(hdc, old_brush)
            win32gui.DeleteObject(brush)

    def _build_wnd_proc(self):
        def _wnd_proc(hwnd, msg, wparam, lparam):
            if msg == win32con.WM_MOUSEMOVE:
                x, y = win32gui.ScreenToClient(hwnd, win32api.GetCursorPos())
                self.world.win_proc("mouse_move", x, y, self.width, self.height)
                win32gui.InvalidateRect(hwnd, None, False)
                return 0
            if msg == win32con.WM_LBUTTONDOWN:
                x, y = win32gui.ScreenToClient(hwnd, win32api.GetCursorPos())
                self.world.win_proc("mouse_click", x, y, self.width, self.height)
                win32gui.InvalidateRect(hwnd, None, False)
                return 0
            if msg == win32con.WM_PAINT:
                hdc, paint_struct = win32gui.BeginPaint(hwnd)
                self._draw_world(hdc)
                win32gui.EndPaint(hwnd, paint_struct)
                return 0
            if msg == win32con.WM_DESTROY:
                win32gui.PostQuitMessage(0)
                return 0
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        return _wnd_proc

    def create_window(self) -> int:
        h_instance = win32api.GetModuleHandle(None)
        wnd_class = win32gui.WNDCLASS()
        wnd_class.hInstance = h_instance
        wnd_class.lpszClassName = self._class_name
        wnd_class.lpfnWndProc = self._wnd_proc
        wnd_class.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        wnd_class.hbrBackground = win32con.COLOR_WINDOW + 1
        win32gui.RegisterClass(wnd_class)

        style = win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE
        self.hwnd = win32gui.CreateWindowEx(
            0,
            self._class_name,
            self.title,
            style,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            self.width,
            self.height,
            0,
            0,
            h_instance,
            None,
        )
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)
        return self.hwnd

    def run(self) -> None:
        if self.hwnd is None:
            self.create_window()
        win32gui.PumpMessages()
