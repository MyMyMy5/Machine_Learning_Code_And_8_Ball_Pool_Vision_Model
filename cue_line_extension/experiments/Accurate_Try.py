import cv2
import numpy as np

def extend_to_rect(p1, p2, rect):
    (x1, y1), (x2, y2) = p1, p2
    xmin, ymin, xmax, ymax = rect
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return p1, p2
    ts = []
    if dx != 0:
        for x in (xmin, xmax):
            t = (x - x1) / dx
            y = y1 + t * dy
            if ymin <= y <= ymax:
                ts.append(t)
    if dy != 0:
        for y in (ymin, ymax):
            t = (y - y1) / dy
            x = x1 + t * dx
            if xmin <= x <= xmax:
                ts.append(t)
    a, b = min(ts), max(ts)
    A = (int(round(x1 + a * dx)), int(round(y1 + a * dy)))
    B = (int(round(x1 + b * dx)), int(round(y1 + b * dy)))
    return A, B

def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["points"].append((x, y))
        cv2.circle(state["overlay"], (x, y), 4, (0, 255, 0), -1)
        if len(state["points"]) == 2:
            A, B = extend_to_rect(state["points"][0], state["points"][1], state["rect"])
            cv2.line(state["overlay"], A, B, (0, 255, 255), 3)
            state["points"].clear()
        cv2.imshow("extend", state["overlay"])

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit("Could not load image")

    # Rough felt detection to get the inner rectangle
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), 2)
    ys, xs = np.where(mask > 0)
    rect = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    overlay = img.copy()
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 1)

    state = {"overlay": overlay, "rect": rect, "points": []}
    cv2.namedWindow("extend", cv2.WINDOW_NORMAL)
    cv2.imshow("extend", overlay)
    cv2.setMouseCallback("extend", on_mouse, state)

    print("Click two points along the guide line. Press 's' to save, 'q' to quit.")
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord('s'):
            cv2.imwrite("extended_by_click.png", state["overlay"])
            print("Saved extended_by_click.png")
        elif k in (ord('q'), 27):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image")
    args = p.parse_args()
    main(args.image)
