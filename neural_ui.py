from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from enum import IntEnum
from string import Template
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised when CuPy is installed
    import cupy as cp
except ImportError:  # pragma: no cover - fallback for environments without CuPy
    cp = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback for environments without NumPy
    np = None

try:  # pragma: no cover - optional dependency for shader rendering
    import glfw
except ImportError:  # pragma: no cover - fallback for environments without GLFW
    glfw = None

try:  # pragma: no cover - optional dependency for shader rendering
    import moderngl
except ImportError:  # pragma: no cover - fallback for environments without ModernGL
    moderngl = None


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
    _world: Optional["NeuralWorld"] = field(default=None, init=False, repr=False, compare=False)
    _suspend_world_sync: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name not in {"x", "y"}:
            return
        object_dict = self.__dict__
        world = object_dict.get("_world")
        if world is None or object_dict.get("_suspend_world_sync", False):
            return
        tensor_index = object_dict.get("tensor_index")
        if tensor_index is None:
            return
        world.set_local_position(tensor_index, float(self.x), float(self.y))

    def _bind_view(self, world: "NeuralWorld", index: int) -> None:
        self._world = world
        self._suspend_world_sync = True
        self.tensor_index = index
        self._suspend_world_sync = False

    def _sync_position_from_world(self, x: float, y: float) -> None:
        self._suspend_world_sync = True
        self.x = float(x)
        self.y = float(y)
        self._suspend_world_sync = False


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
    _ROW_CORNER_RADIUS = 6
    _ROW_BORDER_THICKNESS = 7
    _ROW_SOFTNESS = 8
    _ROW_RESERVED_SHADER_PARAM = 9  # Reserved for future per-instance shader attributes.
    _WORLD_COLUMNS = 10
    _MOTION_COLUMNS = 6
    _MOTION_POSITION_AXES = (_ROW_X, _ROW_Y)
    _MOTION_POSITION_SLICE = slice(_ROW_X, _ROW_Y + 1)
    _COLOR_CHANNELS = 3
    _DEFAULT_CORNER_RADIUS_FACTOR = 0.1
    _DEFAULT_BORDER_THICKNESS = 0.01
    _DEFAULT_SOFTNESS = 0.02
    _SCREEN_BOUNDARY = 1.5
    _REACTIVATE_BOUNDARY = 1.1

    def __init__(
        self,
        use_cupy: bool = True,
        initial_capacity: int = 10_000,
        growth_chunk: int = 10_000,
        friction_coefficient: float = 0.98,
    ) -> None:
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if growth_chunk <= 0:
            raise ValueError("growth_chunk must be positive")
        self._growth_chunk = int(growth_chunk)
        self._capacity = int(initial_capacity)
        self._size = 0
        self._global_dirty = False
        self.friction_coefficient = float(friction_coefficient)
        if self.friction_coefficient < 0.0:
            raise ValueError("friction_coefficient must be non-negative")
        if use_cupy and cp is not None:
            self.xp = cp
            self.backend = "cupy"
            self.world_tensor = self.xp.zeros((self._capacity, self._WORLD_COLUMNS), dtype=self.xp.float32)
            self.velocity_tensor = self.xp.zeros((self._capacity, self._MOTION_COLUMNS), dtype=self.xp.float32)
            self.color_tensor = self.xp.zeros((self._capacity, self._COLOR_CHANNELS), dtype=self.xp.float32)
            self._global_tensor = self.xp.zeros((self._capacity, self._WORLD_COLUMNS), dtype=self.xp.float32)
            self._parent_index = self.xp.full((self._capacity,), -1, dtype=self.xp.int32)
        elif np is not None:
            self.xp = np
            self.backend = "numpy"
            self.world_tensor = self.xp.zeros((self._capacity, self._WORLD_COLUMNS), dtype=self.xp.float32)
            self.velocity_tensor = self.xp.zeros((self._capacity, self._MOTION_COLUMNS), dtype=self.xp.float32)
            self.color_tensor = self.xp.zeros((self._capacity, self._COLOR_CHANNELS), dtype=self.xp.float32)
            self._global_tensor = self.xp.zeros((self._capacity, self._WORLD_COLUMNS), dtype=self.xp.float32)
            self._parent_index = self.xp.full((self._capacity,), -1, dtype=self.xp.int32)
        else:
            self.xp = None
            self.backend = "python"
            self.world_tensor: List[List[float]] = []
            self.velocity_tensor: List[List[float]] = []
            self.color_tensor: List[List[float]] = []
            self._python_global_rows: List[List[float]] = []
            self._python_parent_index: List[int] = []
        self._objects: List[Object] = []
        self._index_to_object: Dict[int, Object] = {}
        self._last_hover_index: Optional[int] = None
        self._default_color: Optional[Tuple[float, float, float]] = None

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
        world_zeros = self.xp.zeros((expand_by, self._WORLD_COLUMNS), dtype=self.xp.float32)
        velocity_zeros = self.xp.zeros((expand_by, self._MOTION_COLUMNS), dtype=self.xp.float32)
        self.world_tensor = self.xp.concatenate((self.world_tensor, world_zeros), axis=0)
        self.velocity_tensor = self.xp.concatenate((self.velocity_tensor, velocity_zeros), axis=0)
        color_zeros = self.xp.zeros((expand_by, self._COLOR_CHANNELS), dtype=self.xp.float32)
        self.color_tensor = self.xp.concatenate((self.color_tensor, color_zeros), axis=0)
        self._global_tensor = self.xp.concatenate((self._global_tensor, world_zeros), axis=0)
        parent_padding = self.xp.full((expand_by,), -1, dtype=self.xp.int32)
        self._parent_index = self.xp.concatenate((self._parent_index, parent_padding), axis=0)
        self._capacity = new_capacity

    def _register_single(self, obj: Object, parent_index: int = -1) -> int:
        corner_radius = min(obj.width, obj.height) * self._DEFAULT_CORNER_RADIUS_FACTOR
        row = [
            obj.x,
            obj.y,
            obj.width / 2.0,
            obj.height / 2.0,
            obj.z,
            float(obj.state),
            corner_radius,
            self._DEFAULT_BORDER_THICKNESS,
            self._DEFAULT_SOFTNESS,
            0.0,
        ]
        if self.backend == "python":
            self.world_tensor.append(row)
            self.velocity_tensor.append([0.0] * self._MOTION_COLUMNS)
            self._python_parent_index.append(parent_index)
            self._python_global_rows.append(row[:])
            index = len(self.world_tensor) - 1
            initial_color = self._initial_color_for_index(index)
            self.color_tensor.append(list(initial_color))
        else:
            index = self._size
            self._ensure_capacity(index + 1)
            row_tensor = self.xp.asarray(row, dtype=self.xp.float32)
            self.world_tensor[index, :] = row_tensor
            self.velocity_tensor[index, :] = 0.0
            initial_color = self._initial_color_for_index(index)
            self.color_tensor[index, :] = self.xp.asarray(initial_color, dtype=self.xp.float32)
            self._parent_index[index] = int(parent_index)
            self._global_tensor[index, :] = row_tensor
            self._size += 1
        obj._bind_view(self, index)
        self._objects.append(obj)
        self._index_to_object[index] = obj
        self._global_dirty = True
        return index

    def set_local_position(self, index: int, x: float, y: float) -> None:
        if index < 0 or index >= self.size:
            raise IndexError("Object index out of bounds")
        if self.backend == "python":
            self.world_tensor[index][self._ROW_X] = float(x)
            self.world_tensor[index][self._ROW_Y] = float(y)
        else:
            self.world_tensor[index, self._ROW_X] = float(x)
            self.world_tensor[index, self._ROW_Y] = float(y)
        obj = self._index_to_object.get(index)
        if obj is not None:
            obj._sync_position_from_world(x, y)
        self._global_dirty = True

    def set_color(self, index: int, color: Sequence[float]) -> None:
        if index < 0 or index >= self.size:
            raise IndexError("Object index out of bounds")
        if len(color) != self._COLOR_CHANNELS:
            raise ValueError("color must contain exactly three channels")
        r, g, b = (float(channel) for channel in color)
        if self.backend == "python":
            self.color_tensor[index] = [r, g, b]
        else:
            self.color_tensor[index, :] = self.xp.asarray([r, g, b], dtype=self.xp.float32)

    def set_default_color(self, color: Sequence[float]) -> None:
        if len(color) != self._COLOR_CHANNELS:
            raise ValueError("color must contain exactly three channels")
        r, g, b = (float(channel) for channel in color)
        self._default_color = (r, g, b)
        if self.size == 0:
            return
        if self.backend == "python":
            for index in range(self.size):
                self.color_tensor[index] = [r, g, b]
            return
        self.color_tensor[: self.size, :] = self.xp.asarray([r, g, b], dtype=self.xp.float32)

    @staticmethod
    def _generate_color_from_index(index: int) -> Tuple[float, float, float]:
        # Linear Congruential Generator constants for deterministic per-index colors.
        seed = (index * 1664525 + 1013904223) & 0xFFFFFFFF
        r = 0.35 + (((seed >> 0) & 0xFF) / 255.0) * 0.65
        g = 0.35 + (((seed >> 8) & 0xFF) / 255.0) * 0.65
        b = 0.35 + (((seed >> 16) & 0xFF) / 255.0) * 0.65
        return r, g, b

    def _initial_color_for_index(self, index: int) -> Tuple[float, float, float]:
        return self._default_color or self._generate_color_from_index(index)

    def update(self, dt: float) -> None:
        # Ignore negligible timesteps to avoid needless tensor work for near-zero frame deltas.
        if abs(float(dt)) < 1e-9:
            return
        if self.backend == "python":
            active_state = float(ObjectState.ACTIVE)
            out_of_screen_state = float(ObjectState.OUTOFSCREEN)
            for idx in range(len(self.world_tensor)):
                row = self.world_tensor[idx]
                vel = self.velocity_tensor[idx]
                row[self._ROW_X] += vel[self._ROW_X] * dt
                row[self._ROW_Y] += vel[self._ROW_Y] * dt
                for axis in self._MOTION_POSITION_AXES:
                    vel[axis] *= self.friction_coefficient
                tracked = row[self._ROW_STATE] in (active_state, out_of_screen_state)
                outside_screen = (
                    abs(row[self._ROW_X]) > self._SCREEN_BOUNDARY or abs(row[self._ROW_Y]) > self._SCREEN_BOUNDARY
                )
                reentered = (
                    abs(row[self._ROW_X]) <= self._REACTIVATE_BOUNDARY
                    and abs(row[self._ROW_Y]) <= self._REACTIVATE_BOUNDARY
                )
                if tracked and outside_screen:
                    row[self._ROW_STATE] = out_of_screen_state
                elif row[self._ROW_STATE] == out_of_screen_state and reentered:
                    row[self._ROW_STATE] = active_state
            self._global_dirty = True
            return

        if self._size == 0:
            return
        rows = self.world_tensor[: self._size]
        velocity = self.velocity_tensor[: self._size]
        rows[:, self._MOTION_POSITION_SLICE] += velocity[:, self._MOTION_POSITION_SLICE] * float(dt)
        velocity[:, self._MOTION_POSITION_SLICE] *= self.friction_coefficient
        self._global_dirty = True
        self._sync_global_transforms()

        global_rows = self._global_tensor[: self._size]
        states = rows[:, self._ROW_STATE]
        x_positions = global_rows[:, self._ROW_X]
        y_positions = global_rows[:, self._ROW_Y]
        abs_x_positions = self.xp.abs(x_positions)
        abs_y_positions = self.xp.abs(y_positions)
        outside = (abs_x_positions > self._SCREEN_BOUNDARY) | (abs_y_positions > self._SCREEN_BOUNDARY)
        inside_reactivate = (abs_x_positions <= self._REACTIVATE_BOUNDARY) & (abs_y_positions <= self._REACTIVATE_BOUNDARY)
        tracked_states = (states == float(ObjectState.ACTIVE)) | (states == float(ObjectState.OUTOFSCREEN))
        states[tracked_states & outside] = float(ObjectState.OUTOFSCREEN)
        states[(states == float(ObjectState.OUTOFSCREEN)) & inside_reactivate] = float(ObjectState.ACTIVE)
        rows[:, self._ROW_STATE] = states
        global_rows[:, self._ROW_STATE] = states

    def _sync_global_transforms(self) -> None:
        if not self._global_dirty:
            return

        if self.backend == "python":
            object_count = len(self.world_tensor)
            if object_count == 0:
                self._global_dirty = False
                return

            for idx, row in enumerate(self.world_tensor):
                self._python_global_rows[idx] = row[:]
                parent_idx = self._python_parent_index[idx]
                if parent_idx < -1:
                    raise ValueError(f"Parent index {parent_idx} for object {idx} is invalid (must be >= -1)")
                if parent_idx >= object_count:
                    raise ValueError(f"Parent index {parent_idx} for object {idx} is out of bounds")

            resolved = [parent_idx == -1 for parent_idx in self._python_parent_index]
            remaining = object_count - sum(resolved)
            while remaining > 0:
                progressed = False
                for idx in range(object_count):
                    if resolved[idx]:
                        continue
                    parent_idx = self._python_parent_index[idx]
                    if parent_idx >= 0 and resolved[parent_idx]:
                        local_row = self.world_tensor[idx]
                        parent_global = self._python_global_rows[parent_idx]
                        global_row = local_row[:]
                        global_row[self._ROW_X] = local_row[self._ROW_X] + parent_global[self._ROW_X]
                        global_row[self._ROW_Y] = local_row[self._ROW_Y] + parent_global[self._ROW_Y]
                        global_row[self._ROW_Z] = local_row[self._ROW_Z] + parent_global[self._ROW_Z]
                        self._python_global_rows[idx] = global_row
                        resolved[idx] = True
                        remaining -= 1
                        progressed = True
                if not progressed:
                    raise ValueError("Hierarchy cycle detected")
            self._global_dirty = False
            return

        if self._size == 0:
            self._global_dirty = False
            return
        active_local = self.world_tensor[: self._size]
        active_global = self._global_tensor[: self._size]
        active_global[:, :] = active_local
        active_parent = self._parent_index[: self._size]
        invalid_parent = (active_parent < -1) | (active_parent >= self._size)
        if bool(self._to_scalar(invalid_parent.any())):
            invalid_indices = self.xp.where(invalid_parent)[0]
            bad_index = int(self._to_scalar(invalid_indices[0]))
            bad_parent = int(self._to_scalar(active_parent[bad_index]))
            raise ValueError(f"Parent index {bad_parent} for object {bad_index} is out of bounds")

        resolved = active_parent == -1
        remaining = self._size - int(self._to_scalar(resolved.sum()))
        while remaining > 0:
            unresolved_indices = self.xp.where(~resolved)[0]
            unresolved_parents = active_parent[unresolved_indices]
            ready_mask = resolved[unresolved_parents]
            ready_indices = unresolved_indices[ready_mask]
            ready_count = int(self._to_scalar(ready_indices.size))
            if ready_count == 0:
                raise ValueError("Hierarchy cycle detected")
            parent_indices = active_parent[ready_indices]
            active_global[ready_indices, self._ROW_X] = (
                active_local[ready_indices, self._ROW_X] + active_global[parent_indices, self._ROW_X]
            )
            active_global[ready_indices, self._ROW_Y] = (
                active_local[ready_indices, self._ROW_Y] + active_global[parent_indices, self._ROW_Y]
            )
            active_global[ready_indices, self._ROW_Z] = (
                active_local[ready_indices, self._ROW_Z] + active_global[parent_indices, self._ROW_Z]
            )
            resolved[ready_indices] = True
            remaining -= ready_count
        self._global_dirty = False

    def sync_global_transforms(self) -> None:
        self._sync_global_transforms()

    def hover_index(self) -> Optional[int]:
        return self._last_hover_index

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


class ModernGLRenderer:
    _VERTEX_FLOATS = 5
    _FLOAT_SIZE_BYTES = 4
    _VERTEX_STRIDE_BYTES = _VERTEX_FLOATS * _FLOAT_SIZE_BYTES
    _INITIAL_VBO_SIZE_BYTES = 8192

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
        vsync: bool = True,
    ) -> None:
        if moderngl is None or glfw is None:
            raise RuntimeError("ModernGLRenderer requires both moderngl and glfw")
        self.world = world
        self.title = title
        self.width = width
        self.height = height
        self.background_color = self._int_to_rgb(background_color)
        self.object_color = self._int_to_rgb(object_color)
        self.hover_color = self._int_to_rgb(hover_color)
        self.vsync = vsync
        self.window = None
        self._ctx = None
        self._program = None
        self._vbo = None
        self._vao = None
        self._left_down = False

    @staticmethod
    def _int_to_rgb(value: int) -> Tuple[float, float, float]:
        return (
            ((value >> 16) & 0xFF) / 255.0,
            ((value >> 8) & 0xFF) / 255.0,
            (value & 0xFF) / 255.0,
        )

    @staticmethod
    def world_to_ndc(
        x_world: float, y_world: float, width_px: int, height_px: int, half_w_world: float, half_h_world: float
    ) -> Tuple[float, float, float, float]:
        aspect = width_px / height_px
        left_ndc = (x_world - half_w_world) / aspect
        right_ndc = (x_world + half_w_world) / aspect
        top_ndc = y_world + half_h_world
        bottom_ndc = y_world - half_h_world
        return left_ndc, top_ndc, right_ndc, bottom_ndc

    def _build_vertices(self) -> bytes:
        vertices = array("f")
        for obj, row in self.world.iter_render_rows():
            if row[NeuralWorld._ROW_STATE] != float(ObjectState.ACTIVE):
                continue
            color = self.hover_color if obj.tensor_index == self.world._last_hover_index else self.object_color
            left, top, right, bottom = self.world_to_ndc(
                row[NeuralWorld._ROW_X],
                row[NeuralWorld._ROW_Y],
                self.width,
                self.height,
                row[NeuralWorld._ROW_HALF_W],
                row[NeuralWorld._ROW_HALF_H],
            )
            r, g, b = color
            vertices.extend((left, top, r, g, b))
            vertices.extend((right, top, r, g, b))
            vertices.extend((right, bottom, r, g, b))
            vertices.extend((left, top, r, g, b))
            vertices.extend((right, bottom, r, g, b))
            vertices.extend((left, bottom, r, g, b))
        return vertices.tobytes()

    def _draw_world(self) -> None:
        self._ctx.clear(*self.background_color, 1.0)
        payload = self._build_vertices()
        if not payload:
            return
        if len(payload) > self._vbo.size:
            self._vbo.orphan(len(payload))
        self._vbo.write(payload)
        self._vao.render(moderngl.TRIANGLES, vertices=len(payload) // self._VERTEX_STRIDE_BYTES)

    def create_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.vsync else 0)

        self._ctx = moderngl.create_context()
        self._program = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(v_color, 1.0);
                }
            """,
        )
        self._vbo = self._ctx.buffer(reserve=self._INITIAL_VBO_SIZE_BYTES)
        self._vao = self._ctx.simple_vertex_array(self._program, self._vbo, "in_pos", "in_color")
        return self.window

    def run(self) -> None:
        if self.window is None:
            self.create_window()
        try:
            while not glfw.window_should_close(self.window):
                mouse_x_px, mouse_y_px = glfw.get_cursor_pos(self.window)
                self.world.win_proc("mouse_move", mouse_x_px, mouse_y_px, self.width, self.height)

                left_pressed = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                if left_pressed and not self._left_down:
                    self.world.win_proc("mouse_click", mouse_x_px, mouse_y_px, self.width, self.height)
                self._left_down = left_pressed

                self._draw_world()
                glfw.swap_buffers(self.window)
                glfw.poll_events()
        finally:
            if self.window is not None:
                glfw.destroy_window(self.window)
                self.window = None
            glfw.terminate()


class InstancedModernGLRenderer(ModernGLRenderer):
    """ModernGL renderer that uses instanced quads for better batching efficiency."""

    _INITIAL_INSTANCE_DATA_BUFFER_SIZE = 20 * 1024
    _INITIAL_INSTANCE_COLOR_BUFFER_SIZE = 12 * 1024
    _SHADER_MIN_SOFTNESS = 0.0005
    _SHADER_GLOW_FALLOFF = 10.0
    _SHADER_GLOW_INTENSITY = 0.35
    _SHADER_SHADOW_OFFSET = 2.0
    _SHADER_SHADOW_FALLOFF = 12.0
    _SHADER_SHADOW_INTENSITY = 0.2
    _SHADER_BORDER_BRIGHTNESS = 0.85
    _SHADER_GLOW_CONTRIBUTION = 0.4
    _SHADER_SHADOW_CONTRIBUTION = 0.2
    _SHADER_GLOW_ALPHA_CONTRIBUTION = 0.75
    _SHADER_ALPHA_DISCARD_THRESHOLD = 0.01
    _INSTANCE_DATA_COLUMNS = (
        NeuralWorld._ROW_X,
        NeuralWorld._ROW_Y,
        NeuralWorld._ROW_HALF_W,
        NeuralWorld._ROW_HALF_H,
        NeuralWorld._ROW_STATE,
        NeuralWorld._ROW_CORNER_RADIUS,
        NeuralWorld._ROW_BORDER_THICKNESS,
        NeuralWorld._ROW_SOFTNESS,
        NeuralWorld._ROW_RESERVED_SHADER_PARAM,
    )

    def __init__(self, *args, random_object_colors: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_object_colors = random_object_colors
        # These are initialized after OpenGL context creation inside create_window().
        self._quad_vbo = None
        self._instance_data_vbo = None
        self._instance_color_vbo = None
        if not self.random_object_colors:
            self.world.set_default_color(self.object_color)

    def _as_cpu_tensor(self, tensor):
        """Convert CuPy tensors to NumPy arrays for byte upload; leave others unchanged."""
        if self.world.backend == "cupy":
            return cp.asnumpy(tensor)
        return tensor

    def _draw_world(self) -> None:
        self._ctx.clear(*self.background_color, 1.0)
        if self.world.backend == "python":
            super()._draw_world()
            return
        if self.world.size == 0:
            return
        self.world.sync_global_transforms()
        rows = self.world._global_tensor[: self.world.size]
        instance_data_tensor = rows[:, self._INSTANCE_DATA_COLUMNS]
        color_tensor = self.world.color_tensor[: self.world.size]
        data_payload = self._as_cpu_tensor(instance_data_tensor).tobytes()
        color_payload = self._as_cpu_tensor(color_tensor).tobytes()
        if not data_payload:
            return
        self._program["aspect"].value = self.width / self.height
        self._program["active_state"].value = float(ObjectState.ACTIVE)
        hover_index = self.world.hover_index()
        self._program["hover_index"].value = -1 if hover_index is None else int(hover_index)
        self._program["hover_color"].value = self.hover_color
        if len(data_payload) > self._instance_data_vbo.size:
            self._instance_data_vbo.orphan(len(data_payload))
        if len(color_payload) > self._instance_color_vbo.size:
            self._instance_color_vbo.orphan(len(color_payload))
        self._instance_data_vbo.write(data_payload)
        self._instance_color_vbo.write(color_payload)
        self._vao.render(moderngl.TRIANGLE_STRIP, instances=self.world.size)

    def create_window(self):
        """Create a GLFW window and configure an instanced quad-rendering pipeline."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.vsync else 0)

        self._ctx = moderngl.create_context()
        self._program = self._ctx.program(
            vertex_shader="""
                #version 330
                uniform float aspect;
                uniform float active_state;
                uniform int hover_index;
                uniform vec3 hover_color;
                in vec2 in_vert;
                in vec4 in_data;
                in float in_state;
                in vec4 in_params;
                in vec3 in_color;
                out vec3 v_color;
                out vec2 v_local_pos;
                out vec2 v_half_size;
                out vec4 v_params;
                void main() {
                    float visible = in_state == active_state ? 1.0 : 0.0;
                    vec2 scale = in_data.zw * visible;
                    vec2 pos = in_data.xy + (in_vert * scale);
                    pos.x /= aspect;
                    gl_Position = vec4(pos, 0.0, 1.0);
                    v_color = gl_InstanceID == hover_index ? hover_color : in_color;
                    v_local_pos = in_vert * in_data.zw;
                    v_half_size = in_data.zw;
                    v_params = in_params;
                }
            """,
            fragment_shader=Template(
                """
                #version 330
                const float MIN_SOFTNESS = $min_softness;
                const float MIN_INNER_SIZE = $min_inner_size;
                const float GLOW_FALLOFF = $glow_falloff;
                const float GLOW_INTENSITY = $glow_intensity;
                const float SHADOW_OFFSET = $shadow_offset;
                const float SHADOW_FALLOFF = $shadow_falloff;
                const float SHADOW_INTENSITY = $shadow_intensity;
                const float BORDER_BRIGHTNESS = $border_brightness;
                const float GLOW_CONTRIBUTION = $glow_contribution;
                const float SHADOW_CONTRIBUTION = $shadow_contribution;
                const float GLOW_ALPHA_CONTRIBUTION = $glow_alpha_contribution;
                const float ALPHA_DISCARD_THRESHOLD = $alpha_discard_threshold;
                in vec3 v_color;
                in vec2 v_local_pos;
                in vec2 v_half_size;
                in vec4 v_params;
                out vec4 fragColor;

                float roundedBoxSdf(vec2 point, vec2 halfSize, float radius) {
                    vec2 q = abs(point) - (halfSize - vec2(radius));
                    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
                }

                void main() {
                    float cornerRadius = clamp(v_params.x, 0.0, min(v_half_size.x, v_half_size.y));
                    float borderThickness = max(v_params.y, 0.0);
                    float softness = max(v_params.z, MIN_SOFTNESS);
                    float sdf = roundedBoxSdf(v_local_pos, v_half_size, cornerRadius);

                    // Prevent degenerate inner geometry when thick borders collapse the fill area.
                    vec2 innerHalfSize = max(v_half_size - vec2(borderThickness), vec2(MIN_INNER_SIZE));
                    float innerRadius = clamp(cornerRadius - borderThickness, 0.0, min(innerHalfSize.x, innerHalfSize.y));
                    float innerSdf = roundedBoxSdf(v_local_pos, innerHalfSize, innerRadius);

                    float fillAlpha = 1.0 - smoothstep(0.0, softness, sdf);
                    float innerAlpha = 1.0 - smoothstep(0.0, softness, innerSdf);
                    float borderAlpha = clamp(fillAlpha - innerAlpha, 0.0, 1.0);

                    float glow = exp(-max(sdf, 0.0) / (softness * GLOW_FALLOFF)) * GLOW_INTENSITY;
                    float shadow = exp(-max(sdf + softness * SHADOW_OFFSET, 0.0) / (softness * SHADOW_FALLOFF)) * SHADOW_INTENSITY;

                    vec3 color = v_color * (innerAlpha + borderAlpha * BORDER_BRIGHTNESS)
                        + v_color * glow * GLOW_CONTRIBUTION
                        - vec3(shadow * SHADOW_CONTRIBUTION);
                    float alpha = max(fillAlpha, glow * GLOW_ALPHA_CONTRIBUTION);
                    if (alpha < ALPHA_DISCARD_THRESHOLD) {
                        discard;
                    }
                    fragColor = vec4(color, clamp(alpha, 0.0, 1.0));
                }
                """
            ).substitute(
                min_softness=self._SHADER_MIN_SOFTNESS,
                min_inner_size=self._SHADER_MIN_SOFTNESS,
                glow_falloff=self._SHADER_GLOW_FALLOFF,
                glow_intensity=self._SHADER_GLOW_INTENSITY,
                shadow_offset=self._SHADER_SHADOW_OFFSET,
                shadow_falloff=self._SHADER_SHADOW_FALLOFF,
                shadow_intensity=self._SHADER_SHADOW_INTENSITY,
                border_brightness=self._SHADER_BORDER_BRIGHTNESS,
                glow_contribution=self._SHADER_GLOW_CONTRIBUTION,
                shadow_contribution=self._SHADER_SHADOW_CONTRIBUTION,
                glow_alpha_contribution=self._SHADER_GLOW_ALPHA_CONTRIBUTION,
                alpha_discard_threshold=self._SHADER_ALPHA_DISCARD_THRESHOLD,
            ),
        )
        quad = array("f", (-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
        self._quad_vbo = self._ctx.buffer(quad.tobytes())
        self._instance_data_vbo = self._ctx.buffer(reserve=self._INITIAL_INSTANCE_DATA_BUFFER_SIZE)
        self._instance_color_vbo = self._ctx.buffer(reserve=self._INITIAL_INSTANCE_COLOR_BUFFER_SIZE)
        self._vao = self._ctx.vertex_array(
            self._program,
            [
                (self._quad_vbo, "2f", "in_vert"),
                (self._instance_data_vbo, "4f 1f 4f /i", "in_data", "in_state", "in_params"),
                (self._instance_color_vbo, "3f /i", "in_color"),
            ],
        )
        return self.window



class Win32Renderer(ModernGLRenderer):
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "Win32Renderer has been replaced by ModernGLRenderer. "
            "Instantiate ModernGLRenderer directly to use shader-based rendering."
        )
