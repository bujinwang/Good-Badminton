"""Generate court_annotations.txt headlessly so main.py needs no GUI clicks.

Usage:
    python prepare_annotation.py templates/demo.png outputs/demo
"""
import os
import sys

import cv2

from badminton_analysis.court.mapper import resolve_court_corners


def main():
    template = sys.argv[1] if len(sys.argv) > 1 else "templates/demo.png"
    save_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/demo"

    image = cv2.imread(template)
    if image is None:
        raise SystemExit(f"Cannot read template image: {template}")

    corners, roi_corners, mid_height = resolve_court_corners(image)
    if not corners:
        raise SystemExit("Auto court detection failed; manual corner picking is required.")
    print("corners:", corners)
    print("roi_corners:", roi_corners)
    print("mid_height:", mid_height)

    os.makedirs(save_dir, exist_ok=True)
    target = os.path.join(save_dir, "court_annotations.txt")
    with open(target, "w") as f:
        f.write(f"corners={corners}\n")
        f.write(f"roi_corners={roi_corners}\n")
        f.write(f"mid_height={mid_height}\n")
    print("wrote", target)


if __name__ == "__main__":
    main()
