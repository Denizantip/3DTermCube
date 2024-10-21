#!/usr/bin/python

import fcntl
import signal
import sys
import termios
import tty
from collections import defaultdict

import numpy as np


class ANSI:
    SHOW_CURSOR = "\033[?25h"
    HIDE_CURSOR = "\033[?25l"
    # Save cursor position:
    SAVE_CURSOR_POSITION = "\033[s"
    # Restore cursor position:
    RESTORE_CURSOR_POSITION = "\033[u"
    CLS = "\033[2J"
    # ENABLE_BUFFER = '\033[?47h'
    ENABLE_BUFFER = '\033[?1049h'
    # DISABLE_BUFFER = '\033[?47l'
    DISABLE_BUFFER = '\033[?1049l'
    HOME = "\033[H"
    SMALL = "\033[=1h"
    BIG = "\033[=3h"
    UNDERSCORE = "\033[4m"
    NORMAL = "\033[0m"


class Term:
    e = "\x1b["
    alt_screen = f"{e}?1049h"
    normal_screen = f"{e}?1049l"
    clear = f"{e}2J{e}00f"
    clear_end = f"{e}0J"
    clear_begin = f"{e}1J"
    # mouse_click_on = f"{e}?1002h{e}?1015h{e}?1006h"  # Enable reporting of mouse position on click and release
    mouse_click_on = f"{e}?1000;1002;1015;1006h"  # Enable reporting of mouse position on click and release
    # mouse_click_off = f"{e}?1002l{e}?1015l{e}?1006l"
    mouse_click_off = f"{e}?1000;1002;1015;1006l"
    mouse_direct_on = f"{e}?1003h"  # Enable reporting of mouse position at any movement
    mouse_direct_off = f"{e}?1003l"
    sync_start = f"{e}?2026h"  # Start of terminal synchronized output
    sync_end = f"{e}?2026l"  # End of terminal synchronized output


class Canvas:
    _pixel_map = ((0x01, 0x08),
                  (0x02, 0x10),
                  (0x04, 0x20),
                  (0x40, 0x80))

    # braille unicode characters starts at 0x2800
    _braille_char_offset = 0x2800

    def __init__(self):
        self.clear()

    def clear(self):
        sys.stdout.write(ANSI.CLS)
        sys.stdout.write(ANSI.HOME)
        sys.stdout.flush()
        self.chars = defaultdict(lambda: defaultdict(int))
        self.resolution = self._get_canvas_size()

    @staticmethod
    def _get_canvas_size():
        terminal_size = np.array([0, 0, 0, 0], dtype=np.ushort)

        fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, terminal_size)
        return terminal_size[1] * 2, terminal_size[0] * 4

    @staticmethod
    def _get_pos(x, y):
        """Convert x, y to cols, rows"""
        return x // 2, y // 4

    def set(self, x, y):
        """Set a pixel of the :class:`Canvas` object.

        :param x: x coordinate of the pixel
        :param y: y coordinate of the pixel
        """
        col, row = self._get_pos(x, y)

        if type(self.chars[row][col]) != int:
            return
        # this where the braile magic happens =)
        self.chars[row][col] |= self._pixel_map[y % 4][x % 2]


class RawTTY:

    def __init__(self, hide=True):
        self.hide = hide
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)

    def __enter__(self):
        sys.stdout.write(Term.mouse_click_on)
        tty.setraw(self.fd)

        sys.stdout.write(ANSI.SAVE_CURSOR_POSITION)
        sys.stdout.write(ANSI.SMALL)
        sys.stdout.write(ANSI.ENABLE_BUFFER)
        sys.stdout.write(ANSI.HOME)
        sys.stdout.write(ANSI.CLS)

        if self.hide:
            sys.stdout.write(ANSI.HIDE_CURSOR)
        sys.stdout.flush()

    def __exit__(self, *args, **kwargs):
        sys.stdout.write(Term.mouse_click_off)
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        sys.stdout.write(ANSI.BIG)
        sys.stdout.write(ANSI.DISABLE_BUFFER)
        sys.stdout.write(ANSI.RESTORE_CURSOR_POSITION)

        if self.hide:
            sys.stdout.write(ANSI.SHOW_CURSOR)
        sys.stdout.flush()


def perspective(fovy, aspect, z_near, z_far):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    perspective_matrix = np.zeros((4, 4))
    perspective_matrix[0, 0] = f / aspect
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = -(z_far + z_near) / (z_far - z_near)
    perspective_matrix[3, 2] = -2.0 * z_far * z_near / (z_far - z_near)
    perspective_matrix[2, 3] = -1.0
    return perspective_matrix


def look_at_translate(eye):
    tr = np.eye(4)
    tr[3, :3] = -eye
    return tr


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def look_at_rotate_rh(eye, center, up):
    forward = normalize(center - eye).ravel()
    right = normalize(np.cross(up, forward)).ravel()
    new_up = np.cross(forward, right)
    rot = np.eye(4)
    rot[:3, :3] = np.column_stack((right, new_up, forward))
    return rot


def view_port(resolution, far=0.1, near=10):
    width, height = resolution
    depth = far - near
    m = np.array(
        [
            [width / 2, 0, 0, 0],  # noqa
            [0, height / 2, 0, 0],  # noqa
            [0, 0, depth / 2, 0],  # noqa
            [width / 2, height / 2, depth / 2, 1],  # noqa
        ]
    )
    return m


def rotate_xyz(a):
    x, y, z = np.deg2rad(a)

    rotate_x = np.array(
        [
            [1, 0, 0, 0],  # noqa
            [0, np.cos(y), -np.sin(y), 0],  # noqa
            [0, np.sin(y), np.cos(y), 0],  # noqa
            [0, 0, 0, 1],  # noqa
        ],
        dtype=np.float32,
    ).T

    rotate_y = np.array(
        [
            [np.cos(x), 0, np.sin(x), 0],  # noqa
            [0, 1, 0, 0],  # noqa
            [-np.sin(x), 0, np.cos(x), 0],  # noqa
            [0, 0, 0, 1],  # noqa
        ],
        dtype=np.float32,
    ).T

    rotate_z = np.array(
        [
            [np.cos(z), np.sin(z), 0, 0],  # noqa
            [-np.sin(z), np.cos(z), 0, 0],  # noqa
            [0, 0, 1, 0],  # noqa
            [0, 0, 0, 1],  # noqa
        ],
        dtype=np.float32,
    ).T

    return rotate_z @ rotate_y @ rotate_x


def bresenham_line(start_point, end_point):
    delta = end_point - start_point
    if delta[0] > 0:
        # Always draw line from right to left. Long story why O_o.
        return bresenham_line(end_point, start_point)
    steps = max(abs(delta[:2]))
    if steps == 0:
        return start_point[None]
    step_size = delta / steps

    return start_point + np.arange(int(steps))[:, None] * step_size


def draw(canvas, angles=np.array([0, 0, 0]), backface_culling=False):
    width, height = canvas.resolution
    Cube.vertices = Cube.vertices @ rotate_xyz(angles)
    vertices = Cube.vertices @ (look_at_translate(np.array((0, 0, 3))) @ look_at_rotate_rh(
        np.array((0, 0, 3)), np.array((0, 0, 0)), np.array((0, 1, 0))
    )) @ perspective(90, width / height, 0.1, 10) @ view_port((width, height))
    vertices /= vertices[..., [3]]

    esc = b'\x1b'

    for face in vertices[Cube.faces]:
        a, b, c, *_ = face[..., :3]
        normal = np.cross(b - a, c - a)
        if normal[2] <= 0 and backface_culling:
            continue

        for i in range(4):
            edge = (face[i], face[(i + 1) % 4])
            pxls = bresenham_line(*edge)
            if normal[2] < 0:
                mask = np.bitwise_and(np.arange(len(pxls)) // 4, 1, dtype=np.int8).view(np.bool_)
                pxls = pxls[mask]
            for x, y, *_ in pxls:
                canvas.set(int(x), int(y))

    for x in canvas.chars:
        for y in canvas.chars[x]:
            sys.stdout.write(f"{esc.decode()}[{x};{y}H")
            sys.stdout.write(f"{chr(canvas._braille_char_offset + canvas.chars[x][y])}")
    sys.stdout.flush()


def main():
    sensitivity = 1
    backface_culling = False
    delta = np.array([0, 0, 0])
    prev_mouse_pos = None
    while True:
        key = sys.stdin.buffer.raw.read(100)

        sys.stdout.write(ANSI.CLS)
        sys.stdout.write(ANSI.HOME)
        sys.stdout.flush()
        canvas.clear()
        esc = b'\x1b'
        sys.stdout.write(f"{esc.decode()}[{0};{0}H")
        sys.stdout.write(f"{ANSI.UNDERSCORE}B{ANSI.NORMAL}ackFaces: {backface_culling}\n\rCTRL+C exit")

        if key in [b'\x03', b'\x1a', b'\x04']:
            break

        if key == b'b':
            backface_culling = not backface_culling

        if key and key[0] == 27 and key[1] == 91:
            button, x, y, *_ = key[3: -1].split(b";")
            pressed = key[-1] == 77
            try:  # if data from stdin read incorrectly
                if pressed:
                    sys.stdout.write(f"{esc.decode()}[{y.decode()};{x.decode()}H")
                    sys.stdout.write(f"█")

                x = int(x.decode())
                y = int(y.decode()) * 2
            except ValueError:
                termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                continue

            if prev_mouse_pos is None:
                prev_mouse_pos = np.array([x, y, 0])
            delta = np.array([x, y, 0]) - prev_mouse_pos
            if pressed:
                prev_mouse_pos = np.array([x, y, 0])
            else:
                prev_mouse_pos = None
        draw(canvas, delta * sensitivity, backface_culling)

        sys.stdout.flush()


def reset_canvas(signum, frame):
    canvas.clear()
    draw(canvas)


class Cube:
    r"""
    Cube in clip space
        A                 B
          o-------------o
         /|            /|            y
        / |           / |            |1
    C  o-------------o D|            |
       |  |          |  |            |_________ x
       |G o----------|--o H         /0        1
       | /           | /           /
       |/            |/           /1
       o-------------o           z
    E                F
                            X,    Y,    Z,   W     """
    vertices = np.array([[-1.0, -1.0, 1.0, 1.0],  # E   0
                         [1.0, -1.0, 1.0, 1.0],  # F   1
                         [-1.0, 1.0, 1.0, 1.0],  # C   2
                         [1.0, 1.0, 1.0, 1.0],  # D   3

                         [-1.0, 1.0, -1.0, 1.0],  # A   4
                         [1.0, 1.0, -1.0, 1.0],  # B   5
                         [-1.0, -1.0, -1.0, 1.0],  # G   6
                         [1.0, -1.0, -1.0, 1.0]  # H   7
                         ])

    # segments =     [(E, F), (F, D), (D, C), (C, E), (B, A), (H, B), (G, H), (A, G), (C, A), (D, B), (F, H), (E, G)]
    edges = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (5, 4), (7, 5), (6, 7), (4, 6), (2, 4), (3, 5), (1, 7), (0, 6)])

    # (C, A, B, D), (E, F, H, G), (E, C, D, F)] # noqa
    faces = np.array([(2, 4, 5, 3), (0, 1, 7, 6), (0, 2, 3, 1),
                      # (B, A, G, H), (D, B, H, F), (A, C, E, G) # noqa
                      (5, 4, 6, 7), (3, 5, 7, 1), (4, 2, 0, 6)
                      ])


# Set the signal handler and a 5-second alarm
signal.signal(signal.SIGWINCH, reset_canvas)

with RawTTY(hide=True):
    canvas = Canvas()
    reset_canvas(None, None)
    main()
