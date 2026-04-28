from __future__ import annotations


def is_rect_imgsz(imgsz):
    return isinstance(imgsz, (list, tuple)) and len(imgsz) == 2


def imgsz_hw(imgsz):
    if is_rect_imgsz(imgsz):
        return int(imgsz[0]), int(imgsz[1])
    x = int(imgsz)
    return x, x


def imgsz_long(imgsz):
    h, w = imgsz_hw(imgsz)
    return max(h, w)


def imgsz_str(imgsz):
    h, w = imgsz_hw(imgsz)
    return f"{h}x{w}" if h != w else str(h)
