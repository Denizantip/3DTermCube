#!/usr/bin/env python3
"""
Interactive 3D Cube Renderer for Terminal

Renders a rotating 3D wireframe cube using Braille Unicode characters.
Supports mouse-based rotation and zoom controls.

Architecture Overview:
---------------------

    ┌─────────────┐
    │  Terminal   │  Raw terminal mode, mouse events, Braille rendering
    │   Control   │
    └──────┬──────┘
           │
    ┌──────▼───────┐
    │  Braille     │  2x4 pixel resolution per character cell
    │   Canvas     │  Maps pixels → Braille dots → Unicode chars
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Renderer   │  3D → 2D transformation pipeline
    │              │  • Model transform (rotation)
    │              │  • View transform (camera)
    │              │  • Projection (perspective)
    │              │  • Viewport (screen mapping)
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Cube Model  │  Vertices, faces, geometry definition
    └──────────────┘

Graphics Pipeline:
-----------------
    Model Space  →  World Space  →  View Space  →  Clip Space  →  Screen Space
    (local 3D)      (global 3D)     (camera 3D)     (projected)    (2D pixels)

Key Features:
------------
- Real-time 3D rendering in terminal using Braille characters
- Mouse drag to rotate (natural direction)
- Mouse wheel / +/- keys to zoom
- Backface culling with dashed hidden lines
- Proper perspective projection
- Minimal dependencies (numpy only)

Controls:
--------
- Mouse Drag      : Rotate cube
- Mouse Wheel/+/- : Zoom in/out
- B               : Toggle backface culling
- Ctrl+C          : Exit
"""

import fcntl
import signal
import sys
import termios
import tty
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

import numpy as np


class Terminal:
    """ANSI escape sequences for terminal control."""

    # Cursor control
    SHOW_CURSOR = "\033[?25h"
    HIDE_CURSOR = "\033[?25l"
    SAVE_POSITION = "\033[s"
    RESTORE_POSITION = "\033[u"
    HOME = "\033[H"

    # Screen control
    CLEAR = "\033[2J"
    ENABLE_ALT_BUFFER = "\033[?1049h"
    DISABLE_ALT_BUFFER = "\033[?1049l"

    # Text formatting
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    # Mouse control
    MOUSE_ON = "\033[?1000;1002;1015;1006h"  # Click + drag + wheel tracking
    MOUSE_OFF = "\033[?1000;1002;1015;1006l"

    @staticmethod
    def move_to(row: int, col: int) -> str:
        """Move cursor to specific position."""
        return f"\033[{row};{col}H"


class RawTerminalMode:
    """Context manager for raw terminal mode with mouse support."""

    def __init__(self, hide_cursor: bool = True):
        self.hide_cursor = hide_cursor
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setraw(self.fd)
        sys.stdout.write(Terminal.ENABLE_ALT_BUFFER)
        sys.stdout.write(Terminal.CLEAR)
        sys.stdout.write(Terminal.HOME)
        sys.stdout.write(Terminal.MOUSE_ON)

        if self.hide_cursor:
            sys.stdout.write(Terminal.HIDE_CURSOR)

        sys.stdout.flush()
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        sys.stdout.write(Terminal.MOUSE_OFF)
        sys.stdout.write(Terminal.DISABLE_ALT_BUFFER)
        sys.stdout.write(Terminal.RESTORE_POSITION)

        if self.hide_cursor:
            sys.stdout.write(Terminal.SHOW_CURSOR)

        sys.stdout.flush()


class BrailleCanvas:
    """Canvas using Braille Unicode characters for 2x4 pixel resolution per cell.

    Each terminal character cell can display a Braille character with 8 dots:

        Terminal Cell       Braille Dots        Bit Values
        ┌─────────┐        ┌───┬───┐          ┌────┬────┐
        │    ⣿    │   →    │ 1 │ 4 │    →     │0x01│0x08│
        └─────────┘        │ 2 │ 5 │          │0x02│0x10│
                           │ 3 │ 6 │          │0x04│0x20│
                           │ 7 │ 8 │          │0x40│0x80│
                           └───┴───┘          └────┴────┘

    Dots are combined with bitwise OR to form complete characters.
    Example: dot 1 + dot 4 = 0x01 | 0x08 = 0x09 = ⠉

    Unicode: Braille patterns start at U+2800 (⠀)
             Adding bits 0-255 gives us ⠀ to ⣿
    """

    # Braille dot patterns (2 columns × 4 rows)
    # ⣿ = all dots filled (0xFF = 255)
    _PIXEL_MAP = np.array([
        [0x01, 0x08],  # Row 1: dots 1, 4
        [0x02, 0x10],  # Row 2: dots 2, 5
        [0x04, 0x20],  # Row 3: dots 3, 6
        [0x40, 0x80],  # Row 4: dots 7, 8
    ])

    _BRAILLE_OFFSET = 0x2800  # Unicode offset for Braille patterns

    def __init__(self):
        self.chars = defaultdict(lambda: defaultdict(int))
        self.width, self.height = self._get_terminal_size()

    @staticmethod
    def _get_terminal_size() -> Tuple[int, int]:
        """Get terminal size and convert to Braille pixel resolution."""
        size = np.array([0, 0, 0, 0], dtype=np.ushort)
        fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, size)
        # Each terminal cell = 2x4 Braille pixels
        return size[1] * 2, size[0] * 4

    def clear(self):
        """Clear the canvas."""
        self.chars.clear()
        self.width, self.height = self._get_terminal_size()

    def set_pixel(self, x: int, y: int):
        """Set a pixel at (x, y) coordinates.

        Coordinate mapping from pixel space to terminal character space:

            Pixel Grid (8x16):              Terminal Grid (4x4 chars):
           ┌─┬─┬─┬─┬─┬─┬─┬─┐               ┌───┬───┬───┬───┐
           │0│1│2│3│4│5│6│7│ y=0           │⠀  │⠀  │⠀  │⠀  │ row=0
           ├─┼─┼─┼─┼─┼─┼─┼─┤               ├───┼───┼───┼───┤
           │ │ │ │ │ │ │ │ │ y=1           │⠀  │⠀  │⠀  │⠀  │ row=1
           ├─┼─┼─┼─┼─┼─┼─┼─┤               ├───┼───┼───┼───┤
           │ │ │ │ │ │ │ │ │ y=2   →       │⠀  │⠀  │⠀  │⠀  │ row=2
           ├─┼─┼─┼─┼─┼─┼─┼─┤               ├───┼───┼───┼───┤
           │ │ │ │ │ │ │ │ │ y=3           │⠀  │⠀  │⠀  │⠀  │ row=3
           └─┴─┴─┴─┴─┴─┴─┴─┘               └───┴───┴───┴───┘
         x: 0 1 2 3 4 5 6 7             col: 0   1   2   3

        Each terminal character = 2×4 pixels
        col = x // 2,  row = y // 4
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        col, row = x // 2, y // 4
        # Use modulo to find which dot within the Braille character (0-1 for x, 0-3 for y)
        self.chars[row][col] |= self._PIXEL_MAP[y % 4, x % 2]

    def render(self):
        """Render the canvas to terminal."""
        for row in sorted(self.chars.keys()):
            for col in sorted(self.chars[row].keys()):
                char_code = self._BRAILLE_OFFSET + self.chars[row][col]
                sys.stdout.write(Terminal.move_to(row + 1, col + 1))  # Terminal is 1-indexed
                sys.stdout.write(chr(char_code))
        sys.stdout.flush()


def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def perspective_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create perspective projection matrix.

    Transforms 3D view space to clip space with perspective foreshortening:

    View Space (3D)              Clip Space (4D homogeneous)
         y                              y clip
         ^                             ^
         │    ╱╲                       │   ┌─────┐
         │   ╱  ╲                      │   │     │
         │  ╱    ╲    near plane       │   │     │  Normalized Device Coords
         │ ╱      ╲                    │   │     │  after perspective divide
         │╱________╲                   │   └─────┘
         └──────────> z                └──────────> x clip
        ╱ ╲        far plane          ╱
       ╱   ╲                          ╱ z clip
      ╱     ╲

    The matrix maps the view frustum (truncated pyramid) to a cube [-1,1]³
    Objects closer to camera appear larger after perspective division by w.
    """
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],  # This -1 in [3,2] creates the perspective division
    ])


def look_at_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create view matrix (look-at transformation).

    Builds a camera transformation that positions the camera at 'eye',
    looking towards 'center', with 'up' defining which way is up.

    World Space          →          View Space (Camera Space)

         y ^                             y ^ (up)
           │                               │
           │    ┌─────┐                    │
           │    │ obj │                    │   🎥 camera at origin
           │    └─────┘                    │   looking down -z
           └────────────> x                └──────────> x (right)
          ╱                               ╱
        ╱ z                             ╱ z (forward)

    We construct an orthonormal basis from three vectors:
    - forward = normalize(center - eye)  # where camera looks
    - right   = normalize(up × forward)  # camera's right direction
    - new_up  = forward × right          # camera's actual up direction

    Then combine rotation (change of basis) + translation (move to eye).
    """
    forward = normalize(center - eye)
    right = normalize(np.cross(up, forward))
    new_up = np.cross(forward, right)

    # Combine rotation and translation
    rotation = np.eye(4)
    rotation[:3, :3] = np.column_stack([right, new_up, forward])

    translation = np.eye(4)
    translation[3, :3] = -eye  # Move world so camera is at origin

    return translation @ rotation


def rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Create rotation matrix from Euler angles (degrees).

    Combines three rotation matrices for X, Y, and Z axes.
    Each rotation happens around one axis while keeping the other two fixed.

    Rotation visualizations:

    X-axis (pitch):          Y-axis (yaw):           Z-axis (roll):
         y                        z                       y
         ^                       ^                        ^
         │    ╱                  │                   ╱    │    ╲
         │  ╱z                   │    ╱x           ╱      │      ╲
         │╱___> x                │  ╱            ╱        │        ╲
        rotation around x        │╱____> y     ╱__________│__________╲x
                               rotation          rotation around z
                               around y

    Combined order: Rz @ Ry @ Rx (intrinsic rotations)
    This means: first rotate around X, then Y, then Z in the local frame.
    """
    rx, ry, rz = np.radians(angles)

    # Rotation around X axis (pitch - looking up/down)
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1],
    ])

    # Rotation around Y axis (yaw - looking left/right)
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1],
    ])

    # Rotation around Z axis (roll - tilting head)
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    return Rz @ Ry @ Rx


def bresenham_line(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Generate points along a line using Bresenham's algorithm.

    This is a classic rasterization algorithm that draws a straight line
    between two points using only integer arithmetic (fast!).

    Example: Drawing from (1,1) to (5,3)

        y  ^
        3  │     ·  ·  ●        ← end point (5,3)
        2  │  ·  ●  ●
        1  │  ●  ●              ← start point (1,1)
        0  │
           └───────────────> x
             0  1  2  3  4  5

    Algorithm: Calculate how many steps needed (max of Δx, Δy),
               then interpolate along the line by that step size.
    """
    delta = p2 - p1
    steps = int(max(abs(delta[:2])))  # Use x or y, whichever is larger

    if steps == 0:
        return p1[np.newaxis, :]  # Single point

    step_size = delta / steps
    return p1 + np.arange(steps)[:, np.newaxis] * step_size


@dataclass
class Cube:
    """3D Cube model definition.

    Cube vertices in homogeneous coordinates (x, y, z, w):

           y ^
             │
        A────┼────B         Vertices are positioned in a 2×2×2 cube
        │\   │   /│         centered at origin (0, 0, 0).
        │ \  │  / │
        │  C─┼─D  │         Front face (z=+1): E, F, C, D
        │   \│/   │         Back face  (z=-1): G, H, A, B
        G────┴────H
             └────────> x   w=1 for all vertices (homogeneous coords)
            ╱
           ╱ z
          ↓

    Vertex labeling (looking from +z towards origin):

        Front Face (z=1)          Back Face (z=-1)

        C ────────── D            A ────────── B
        │            │            │            │
        │            │            │            │
        │            │            │            │
        E ────────── F            G ────────── H

    This gives us a right-handed coordinate system where:
    - +X = right,  -X = left
    - +Y = up,     -Y = down
    - +Z = front,  -Z = back
    """

    # Vertices in homogeneous coordinates [x, y, z, w]
    VERTICES = np.array([
        #    X     Y    Z   W     Label  Index
        [-1.0, -1.0, 1.0, 1.0],  # E      0  (front-bottom-left)
        [1.0, -1.0, 1.0, 1.0],  # F      1  (front-bottom-right)
        [-1.0, 1.0, 1.0, 1.0],  # C      2  (front-top-left)
        [1.0, 1.0, 1.0, 1.0],  # D      3  (front-top-right)
        [-1.0, 1.0, -1.0, 1.0],  # A      4  (back-top-left)
        [1.0, 1.0, -1.0, 1.0],  # B      5  (back-top-right)
        [-1.0, -1.0, -1.0, 1.0],  # G      6  (back-bottom-left)
        [1.0, -1.0, -1.0, 1.0],  # H      7  (back-bottom-right)
    ], dtype=np.float32)

    # Face definitions - vertex indices in counter-clockwise order (when viewed from outside)
    # This winding order is crucial for backface culling to work correctly!
    #
    # Normal vector points outward when vertices are ordered counter-clockwise:
    #        v1
    #        /\          normal = (v2-v1) × (v3-v1)
    #       /  \         points OUT of the screen
    #      /    \
    #    v2 ───> v3
    #
    FACES = np.array([
        [2, 4, 5, 3],  # Top face    (C, A, B, D) - looking down from +Y
        [0, 1, 7, 6],  # Bottom face (E, F, H, G) - looking up from -Y
        [0, 2, 3, 1],  # Front face  (E, C, D, F) - looking from +Z
        [5, 4, 6, 7],  # Back face   (B, A, G, H) - looking from -Z
        [3, 5, 7, 1],  # Right face  (D, B, H, F) - looking from +X
        [4, 2, 0, 6],  # Left face   (A, C, E, G) - looking from -X
    ])


class CubeRenderer:
    """Renders a 3D cube with rotation and backface culling."""

    def __init__(self, canvas: BrailleCanvas):
        self.canvas = canvas
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.backface_culling = False
        self.camera_distance = 5.0  # Default camera distance

    def render(self):
        """Render the cube to the canvas.

        Graphics Pipeline:

        1. Model Space → World Space (model matrix)
           - Apply rotations around X, Y, Z axes
           - Cube spins in place at origin

        2. World Space → View Space (view matrix)
           - Position camera at distance looking at origin
           - Objects are now relative to camera

        3. View Space → Clip Space (projection matrix)
           - Apply perspective transformation
           - Creates the pyramid viewing frustum

        4. Clip Space → NDC (Normalized Device Coords)
           - Perspective division (divide by w)
           - Maps to cube [-1, 1]³

        5. NDC → Screen Space (viewport transform)
           - Map [-1, 1] to [0, width/height] in pixels
           - Ready to draw on canvas!

        Visual representation of the pipeline:

        Model    >   World   >   View    >   Clip    >   Screen
        ┌───┐       ┌───┐       ┌───┐         ╱╲          ┌─────┐
        │ □ │   >   │ □ │   >   │ □ │   >    ╱  ╲    >    │ □□  │
        └───┘       └───┘       └───┘       ╱____╲        │ □□  │
         3D          3D          3D       Frustum         └─────┘
        (local)    (global)   (camera)   (projected)      2D pixels
        """
        # Build transformation pipeline
        model = rotation_matrix(self.rotation)
        view = look_at_matrix(
            eye=np.array([0, 0, self.camera_distance]),  # Use dynamic camera distance
            center=np.array([0, 0, 0]),
            up=np.array([0, 1, 0])
        )
        projection = perspective_matrix(
            fov=90,
            aspect=self.canvas.width / self.canvas.height,
            near=0.1,
            far=10
        )

        # Transform vertices through model, view, and projection
        vertices = Cube.VERTICES @ model @ view @ projection

        # Perspective division: divide x,y,z by w to get actual screen position
        # Objects further away have larger w, making them appear smaller
        vertices = vertices / vertices[:, [3]]

        # Apply viewport transformation manually (NDC [-1,1] to screen coordinates)
        # This maps the normalized cube to actual pixel coordinates
        vertices[:, 0] = (vertices[:, 0] + 1) * self.canvas.width / 2
        vertices[:, 1] = (vertices[:, 1] + 1) * self.canvas.height / 2

        # Render faces
        for face_indices in Cube.FACES:
            face = vertices[face_indices]

            # Calculate face normal for backface culling
            # Using cross product of two edges:
            #
            #        a
            #       /\          normal = (b-a) × (c-a)
            #      /  \
            #     / △  \        If normal.z > 0, face points toward camera
            #    /______\       If normal.z ≤ 0, face points away (back-facing)
            #   c        b
            #
            a, b, c = face[:3, :3]
            normal = np.cross(b - a, c - a)

            # Skip back-facing polygons if culling is enabled
            if self.backface_culling and normal[2] <= 0:
                continue

            # Draw edges of the face
            is_backface = normal[2] <= 0
            num_vertices = len(face)

            # Draw edges in a loop: v0→v1, v1→v2, v2→v3, v3→v0
            for i in range(num_vertices):
                edge_start = face[i]
                edge_end = face[(i + 1) % num_vertices]

                # Draw the edge (dashed if it's a back-facing edge)
                self._draw_line(edge_start, edge_end, is_backface)

    def _draw_line(self, p1: np.ndarray, p2: np.ndarray, dashed: bool = False):
        """Draw a line on the canvas, optionally dashed for hidden edges.

        Line rendering with hidden edge visualization:

        Solid line (front edge):       Dashed line (back edge):
        ●─●─●─●─●─●─●─●─●─●            ●─●─●   ●─●─●   ●─●─●

        Dashing pattern: Draw every other group of 4 pixels
        Pattern:  [●●●●____●●●●____●●●●____]
        Indices:   0123 4567 891011 ...
        Keep:      ✓✓✓✓ ✗✗✗✗ ✓✓✓✓ ✗✗✗✗

        This creates the visual effect that back edges are "behind" the cube.
        """
        points = bresenham_line(p1, p2)

        # Create dashed pattern for hidden edges
        if dashed:
            # Group pixels into sets of 4, alternate between showing and hiding
            mask = (np.arange(len(points)) // 4) % 2 == 0
            points = points[mask]

        # Draw pixels on the canvas
        for x, y in points[:, :2]:
            self.canvas.set_pixel(int(x), int(y))

    def rotate(self, delta: np.ndarray):
        """Apply rotation delta."""
        self.rotation += delta

    def zoom(self, delta: float):
        """Adjust camera distance (zoom in/out)."""
        self.camera_distance = np.clip(self.camera_distance + delta, 2.0, 15.0)


@dataclass
class MouseState:
    """Track mouse state for drag operations."""
    pressed: bool = False
    last_pos: Tuple[int, int] = None


class InputHandler:
    """Handle keyboard and mouse input."""

    def __init__(self):
        self.mouse = MouseState()
        self.running = True

    def process_input(self) -> Tuple[np.ndarray, bool, float]:
        """
        Process input and return (rotation_delta, toggle_culling, zoom_delta).
        Returns None for rotation_delta if no mouse movement.
        """
        # Set non-blocking mode with a short timeout to allow responsive resizing
        import select

        # Check if input is available (with 50ms timeout)
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)

        if not readable:
            # No input available, return zeros
            return np.array([0.0, 0.0, 0.0]), False, 0.0

        data = sys.stdin.buffer.raw.read(100)
        rotation_delta = np.array([0.0, 0.0, 0.0])
        toggle_culling = False
        zoom_delta = 0.0

        # Handle keyboard input
        if data in [b'\x03', b'\x1a', b'\x04']:  # Ctrl+C, Ctrl+Z, Ctrl+D
            self.running = False
            return rotation_delta, toggle_culling, zoom_delta

        if data.lower() == b'b':
            toggle_culling = True

        # Zoom with +/- or =/- keys
        if data in [b'+', b'=']:
            zoom_delta = -0.5  # Zoom in
        elif data == b'-':
            zoom_delta = 0.5  # Zoom out

        # Handle mouse input
        # Mouse events arrive as escape sequences in SGR format:
        # ESC[<button;x;yM  (mouse pressed)
        # ESC[<button;x;ym  (mouse released)
        #
        # Button codes:
        #   0  = left click
        #   32 = left drag (motion with button held)
        #   64 = scroll wheel up
        #   65 = scroll wheel down
        #
        if len(data) > 3 and data[0] == 27 and data[1] == 91:
            try:
                # Check for SGR mouse mode format: ESC[<button;x;yM/m
                if data[2] == 60:  # '<' indicates SGR format
                    parts = data[3:-1].split(b';')
                    if len(parts) >= 3:
                        button = int(parts[0])
                        x = int(parts[1])
                        y = int(parts[2]) * 2  # Scale for Braille resolution
                        pressed = data[-1] == 77  # 'M' for press, 'm' for release

                        # Mouse wheel
                        if button == 64:  # Scroll up - zoom in
                            zoom_delta = -0.3
                        elif button == 65:  # Scroll down - zoom out
                            zoom_delta = 0.3
                        # Mouse drag (button 0 = left click)
                        elif button == 0 or button == 32:  # 0=click, 32=drag
                            if pressed or button == 32:  # Pressed or dragging
                                if self.mouse.last_pos:
                                    dx = x - self.mouse.last_pos[0]
                                    dy = y - self.mouse.last_pos[1]
                                    # Invert for natural rotation direction
                                    rotation_delta = np.array([-dy, -dx, 0.0])
                                self.mouse.last_pos = (x, y)
                                self.mouse.pressed = True
                            else:  # Released
                                self.mouse.pressed = False
                                self.mouse.last_pos = None
            except (ValueError, IndexError):
                # Clear buffer on malformed input
                termios.tcflush(sys.stdin, termios.TCIOFLUSH)

        return rotation_delta, toggle_culling, zoom_delta


class CubeApp:
    """Main application controller."""

    def __init__(self):
        self.canvas = BrailleCanvas()
        self.renderer = CubeRenderer(self.canvas)
        self.renderer.rotation = np.array([20.0, 30.0, 0.0])  # Initial rotation for better view
        self.input_handler = InputHandler()
        self.needs_render = True  # Flag to ensure initial render

        # Setup signal handler for terminal resize
        signal.signal(signal.SIGWINCH, self._on_resize)

    def _on_resize(self, signum, frame):
        """Handle terminal resize event."""
        self.canvas.clear()
        self.needs_render = True

    def _draw_ui(self):
        """Draw UI overlay."""
        sys.stdout.write(Terminal.move_to(0, 0))
        sys.stdout.write(
            f"{Terminal.BOLD}[B]{Terminal.RESET} Backface Culling: "
            f"{'ON' if self.renderer.backface_culling else 'OFF'}  "
            f"{Terminal.BOLD}[+/-]{Terminal.RESET} Zoom  "
            f"{Terminal.BOLD}[Ctrl+C]{Terminal.RESET} Exit"
        )
        sys.stdout.flush()

    def run(self):
        """Main application loop."""
        with RawTerminalMode(hide_cursor=True):
            # Initial render
            sys.stdout.write(Terminal.CLEAR)
            sys.stdout.write(Terminal.HOME)
            self.renderer.render()
            self.canvas.render()
            self._draw_ui()

            while self.input_handler.running:
                # Process input
                rotation_delta, toggle_culling, zoom_delta = self.input_handler.process_input()

                # Update state based on input
                if toggle_culling:
                    self.renderer.backface_culling = not self.renderer.backface_culling
                    self.needs_render = True

                if np.any(rotation_delta):
                    self.renderer.rotate(rotation_delta)
                    self.needs_render = True

                if zoom_delta != 0:
                    self.renderer.zoom(zoom_delta)
                    self.needs_render = True

                # Render if anything changed (including window resize via signal handler)
                if self.needs_render:
                    # Clear screen
                    sys.stdout.write(Terminal.CLEAR)
                    sys.stdout.write(Terminal.HOME)
                    self.canvas.clear()

                    # Render
                    self.renderer.render()
                    self.canvas.render()
                    self._draw_ui()

                    self.needs_render = False


def main():
    """Application entry point."""
    try:
        app = CubeApp()
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Ensure terminal is restored on error
        sys.stdout.write(Terminal.DISABLE_ALT_BUFFER)
        sys.stdout.write(Terminal.SHOW_CURSOR)
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    main()