from pathlib import Path

import cv2
import numpy as np


# Prepare the region mask and compare Sobel edges with Canny edges.
def prepare_edges(image, name, white_min):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)

    sobel_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    if name == "sudoku":
        # Lower thresholds retain the faint internal grid lines.
        canny = cv2.Canny(smooth, 20, 60, L2gradient=True)
        _, sobel_edges = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)

        # Manually select the puzzle area to exclude the surrounding newspaper.
        # This mask only selects existing edges; its border is not an edge input.
        height, width = gray.shape
        corners = np.array([
            [0.11 * width, 0.14 * height],
            [0.90 * width, 0.11 * height],
            [0.95 * width, 0.94 * height],
            [0.045 * width, 0.94 * height],
        ], dtype=np.int32)
        region_mask = np.zeros_like(gray)
        cv2.fillPoly(region_mask, [corners], 255)
    else:
        canny = cv2.Canny(smooth, 40, 120, L2gradient=True)
        _, sobel_edges = cv2.threshold(magnitude, 80, 255, cv2.THRESH_BINARY)

        # White paint has high brightness and low colour saturation in HSV.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        region_mask = cv2.inRange(hsv, (0, 0, white_min), (179, 70, 255))
        # Include edge pixels just outside the white region as well.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        region_mask = cv2.dilate(region_mask, kernel)

    sobel_edges = cv2.bitwise_and(sobel_edges.astype(np.uint8), region_mask)
    # Sobel is saved for comparison; the thinner Canny edges supply Hough votes.
    hough_edges = cv2.bitwise_and(canny, region_mask)
    return {
        "grayscale": gray,
        "region_mask": region_mask,
        "sobel_magnitude": np.clip(magnitude, 0, 255).astype(np.uint8),
        "sobel_edges": sobel_edges,
        "canny_edges": canny,
        "hough_edges": hough_edges,
    }


# Keep relevant line directions and remove nearly identical Hough detections.
def select_lines(lines, name, rho_gap, angle_gap):
    selected = []
    angle_gap = np.deg2rad(angle_gap)

    # HoughLines can return None when no line receives enough votes.
    if lines is None:
        return selected

    for rho, theta in lines[:, 0]:
        rho, theta = float(rho), float(theta)
        angle = np.rad2deg(theta)

        # Theta describes the line's normal, not the direction of the line.
        # In p2, keep the upright parking dividers rather than horizontal clutter.
        if name == "p2" and not (angle < 35 or angle > 145):
            continue
        # Sudoku grid lines are approximately horizontal or vertical.
        if name == "sudoku" and not (
            angle < 12 or angle > 168 or abs(angle - 90) < 12
        ):
            continue

        duplicate = False
        for previous_rho, previous_theta in selected:
            difference = abs(theta - previous_theta)
            same_direction = (
                abs(rho - previous_rho) < rho_gap and difference < angle_gap
            )
            # Normals near 0 and 180 degrees can describe the same line.
            opposite_direction = (
                abs(rho + previous_rho) < rho_gap
                and np.pi - difference < angle_gap
            )
            if same_direction or opposite_direction:
                duplicate = True
                break

        if not duplicate:
            selected.append((rho, theta))

    return selected


# Convert a line from rho and theta to two endpoints for cv2.line().
def line_endpoints(rho, theta, length):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b

    # (a, b) is the normal; (-b, a) points along the line.
    point1 = (round(x0 - length * b), round(y0 + length * a))
    point2 = (round(x0 + length * b), round(y0 - length * a))
    return point1, point2


# Draw the detected lines in red over a copy of the original image.
def draw_lines(image, lines):
    if image.ndim == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    length = 2 * max(image.shape[:2])
    for rho, theta in lines:
        point1, point2 = line_endpoints(rho, theta, length)
        # HoughLines gives full lines, so they extend beyond the visible paint.
        cv2.line(result, point1, point2, (0, 0, 255), 1, cv2.LINE_AA)
    return result


# Save an image and report an error if saving fails.
def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


# Save six labelled images to show the main processing steps together.
def save_comparison(images, titles, path):
    height, width = images[0].shape[:2]
    title_height = 40
    panel = np.full((2 * (height + title_height), 3 * width, 3), 255, np.uint8)

    for i, (image, title) in enumerate(zip(images, titles)):
        x = (i % 3) * width
        y = (i // 3) * (height + title_height)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        panel[y + title_height:y + title_height + height, x:x + width] = image
        cv2.putText(
            panel, title, (x + 8, y + 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 1, cv2.LINE_AA
        )

    save_image(path, panel)


# Process the three parking images and the Sudoku image, then save the results.
def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs" / "exercise_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Settings were selected by inspecting the masks and line overlays.
    # white_min is the minimum HSV brightness; it is unused for Sudoku.
    # rho_gap (pixels) and angle_gap (degrees) control duplicate removal.
    settings = {
        "p1.jpg": {"white_min": 190, "votes": 65, "rho_gap": 15, "angle_gap": 4},
        "p2.jpg": {"white_min": 200, "votes": 60, "rho_gap": 12, "angle_gap": 3},
        "p3.jpg": {"white_min": 200, "votes": 100, "rho_gap": 8, "angle_gap": 3},
        "sudoku.png": {"white_min": 0, "votes": 140, "rho_gap": 15, "angle_gap": 4},
    }
    summary = ["image,vote_threshold,raw_lines,selected_lines"]

    for filename, parameters in settings.items():
        name = Path(filename).stem
        image_path = project_dir / "assignment-2-data" / filename
        mode = cv2.IMREAD_GRAYSCALE if name == "sudoku" else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), mode)
        if image is None:
            raise FileNotFoundError(f"Could not read {image_path}")

        image_dir = output_dir / name
        image_dir.mkdir(exist_ok=True)
        stages = prepare_edges(image, name, parameters["white_min"])
        # Accumulator resolution: 1 pixel in rho and 1 degree in theta.
        lines = cv2.HoughLines(
            stages["hough_edges"], 1, np.pi / 180, parameters["votes"]
        )
        selected = select_lines(
            lines, name, parameters["rho_gap"], parameters["angle_gap"]
        )
        raw_lines = [] if lines is None else lines[:, 0]
        result = draw_lines(image, selected)

        save_image(image_dir / "original.png", image)
        for stage_name, stage_image in stages.items():
            save_image(image_dir / f"{stage_name}.png", stage_image)
        save_image(image_dir / "all_hough_lines.png", draw_lines(image, raw_lines))
        save_image(image_dir / "detected_lines.png", result)
        save_comparison(
            [image, stages["region_mask"], stages["sobel_edges"],
             stages["canny_edges"], stages["hough_edges"], result],
            ["Original", "Region mask", "Sobel + region mask",
             "Canny", "Canny + region mask", f"Detected lines: {len(selected)}"],
            image_dir / "comparison.png"
        )

        # Store the detected parameters so the overlays can be checked later.
        line_table = ["rho_pixels,theta_degrees"]
        for rho, theta in selected:
            line_table.append(f"{rho:.4f},{np.rad2deg(theta):.4f}")
        (image_dir / "line_parameters.csv").write_text(
            "\n".join(line_table) + "\n", encoding="utf-8"
        )
        summary.append(
            f"{filename},{parameters['votes']},{len(raw_lines)},{len(selected)}"
        )
        print(f"{filename}: {len(raw_lines)} Hough lines, {len(selected)} selected")

    (output_dir / "line_counts.csv").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
