from pathlib import Path

import cv2
import numpy as np


# Calculate the Harris response from a structure matrix at each pixel.
def calculate_harris_response(gray, block_size, k):
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Without normalization, a box filter sums the values in each neighbourhood.
    # These sums form the entries of the 2 x 2 structure matrix.
    sum_xx = cv2.boxFilter(
        gradient_x ** 2, -1, (block_size, block_size), normalize=False
    )
    sum_yy = cv2.boxFilter(
        gradient_y ** 2, -1, (block_size, block_size), normalize=False
    )
    sum_xy = cv2.boxFilter(
        gradient_x * gradient_y, -1, (block_size, block_size), normalize=False
    )
    # Sobel and boxFilter use reflected image borders by default.
    height, width = gray.shape
    response = np.zeros((height, width), dtype=float)
    lambda_small = np.zeros((height, width), dtype=float)
    lambda_large = np.zeros((height, width), dtype=float)

    for y in range(height):
        for x in range(width):
            matrix = np.array([
                [sum_xx[y, x], sum_xy[y, x]],
                [sum_xy[y, x], sum_yy[y, x]],
            ])
            # eigvalsh is for symmetric matrices and returns sorted eigenvalues.
            lambda1, lambda2 = np.linalg.eigvalsh(matrix)
            lambda_small[y, x] = lambda1
            lambda_large[y, x] = lambda2
            response[y, x] = lambda1 * lambda2 - k * (lambda1 + lambda2) ** 2

    return response, gradient_x, gradient_y, lambda_small, lambda_large


# Select strong local maxima and return the image with red corner markers.
def harris_corner_detector(image, block_size=5, threshold=0.02, k=0.04):
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer of at least 3.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be a fraction between 0 and 1.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image.copy()
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    response, gradient_x, gradient_y, lambda_small, lambda_large = (
        calculate_harris_response(gray, block_size, k)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_response = cv2.dilate(response, kernel)
    local_maximum = response == dilated_response
    above_threshold = response > threshold * response.max()
    corners = local_maximum & above_threshold

    # The equality check follows the assignment; equal neighbouring peaks remain.
    rows, cols = np.where(corners)
    for y, x in zip(rows, cols):
        cv2.circle(result, (int(x), int(y)), 3, (0, 0, 255), 1, cv2.LINE_AA)

    stages = {
        "grayscale": gray,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "lambda_small": lambda_small,
        "lambda_large": lambda_large,
        "response": response,
        "threshold_mask": above_threshold.astype(np.uint8) * 255,
        "corner_mask": corners.astype(np.uint8) * 255,
    }
    return result, stages


# Scale a non-negative array to 0-255 for display, without changing calculations.
def scale_for_display(values):
    scaled = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX)
    return scaled.astype(np.uint8)


# Save an image and report an error if saving fails.
def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


# Arrange equally sized images in a labelled grid for the report.
def save_comparison(images, titles, path, columns=3):
    height, width = images[0].shape[:2]
    title_height = 45
    rows = (len(images) + columns - 1) // columns
    panel = np.full(
        (rows * (height + title_height), columns * width, 3), 255, np.uint8
    )

    for i, (image, title) in enumerate(zip(images, titles)):
        x = (i % columns) * width
        y = (i // columns) * (height + title_height)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        panel[y + title_height:y + title_height + height, x:x + width] = image
        cv2.putText(
            panel, title, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 0), 1, cv2.LINE_AA
        )

    save_image(path, panel)


# Compare a few block sizes and thresholds, then save a representative result.
def main():
    project_dir = Path(__file__).resolve().parent
    image_path = project_dir / "assignment-2-data" / "building.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    output_dir = project_dir / "outputs" / "exercise_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(output_dir / "original.png", image)

    # Vary one parameter at a time, keeping the Harris constant k fixed.
    # The main example uses block_size=5 and threshold=0.02.
    settings = [(3, 0.02), (5, 0.01), (5, 0.02), (5, 0.05), (7, 0.02)]
    k = 0.04
    block_images, block_titles = [], []
    threshold_images, threshold_titles = [], []
    measurements = ["block_size,threshold,k,threshold_pixels,corner_count"]

    for block_size, threshold in settings:
        result, stages = harris_corner_detector(image, block_size, threshold, k)
        name = f"block_{block_size}_threshold_{threshold:.2f}"
        setting_dir = output_dir / name
        setting_dir.mkdir(exist_ok=True)
        count = int(np.count_nonzero(stages["corner_mask"]))
        threshold_count = int(np.count_nonzero(stages["threshold_mask"]))

        # Keep the full signed response for calculations. Display its positive
        # part, since positive values are the candidates for corner detection.
        response_display = scale_for_display(np.maximum(stages["response"], 0))
        save_image(setting_dir / "positive_response.png", response_display)
        save_image(setting_dir / "threshold_mask.png", stages["threshold_mask"])
        save_image(setting_dir / "corner_mask.png", stages["corner_mask"])
        save_image(setting_dir / "detected_corners.png", result)

        measurements.append(f"{block_size},{threshold},{k},{threshold_count},{count}")
        title = f"Block {block_size} x {block_size}, threshold {threshold}: {count} corners"
        print(title, flush=True)
        if threshold == 0.02:
            block_images.append(result)
            block_titles.append(title)
        if block_size == 5:
            threshold_images.append(result)
            threshold_titles.append(title)

        if block_size == 5 and threshold == 0.02:
            gradient_x_display = scale_for_display(np.abs(stages["gradient_x"]))
            gradient_y_display = scale_for_display(np.abs(stages["gradient_y"]))
            save_image(output_dir / "grayscale.png", stages["grayscale"])
            save_image(output_dir / "gradient_x.png", gradient_x_display)
            save_image(output_dir / "gradient_y.png", gradient_y_display)
            save_image(
                output_dir / "lambda_small.png",
                scale_for_display(np.maximum(stages["lambda_small"], 0))
            )
            save_image(
                output_dir / "lambda_large.png",
                scale_for_display(np.maximum(stages["lambda_large"], 0))
            )
            save_image(output_dir / "detected_corners.png", result)
            save_comparison(
                [image, gradient_x_display, gradient_y_display,
                 response_display, stages["corner_mask"], result],
                ["Original", "Absolute Sobel x (scaled)", "Absolute Sobel y (scaled)",
                 "Positive Harris response (scaled)", "Corners after 3 x 3 local maxima",
                 f"Detected corners: {count}"],
                output_dir / "processing_steps.png"
            )

            # Record the actual corner locations, not the pixels of the circles.
            coordinates = ["x,y,harris_response"]
            rows, cols = np.where(stages["corner_mask"] > 0)
            for y, x in zip(rows, cols):
                coordinates.append(f"{x},{y},{stages['response'][y, x]:.6e}")
            (output_dir / "corner_coordinates.csv").write_text(
                "\n".join(coordinates) + "\n", encoding="utf-8"
            )

    save_comparison(
        block_images, block_titles, output_dir / "block_size_comparison.png"
    )
    save_comparison(
        threshold_images, threshold_titles, output_dir / "threshold_comparison.png"
    )
    (output_dir / "corner_counts.csv").write_text(
        "\n".join(measurements) + "\n", encoding="utf-8"
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
