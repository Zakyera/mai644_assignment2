from pathlib import Path

import cv2
import numpy as np


# Add salt-and-pepper noise at randomly chosen pixel positions.
def add_salt_and_pepper_noise(image, amount):
    """Set the requested fraction of pixels to black or white."""
    if not 0 <= amount <= 1:
        raise ValueError("The noise amount must be between 0 and 1.")

    noisy = image.copy()
    height, width = image.shape[:2]
    number_noisy = int(amount * height * width)
    number_salt = number_noisy // 2
    number_pepper = number_noisy - number_salt

    # Choose different positions so a pixel cannot receive both types of noise.
    positions = np.random.choice(height * width, number_noisy, replace=False)
    rows = positions // width
    cols = positions % width
    noisy[rows[:number_salt], cols[:number_salt]] = 255
    noisy[rows[number_salt:], cols[number_salt:]] = 0

    # If the total is odd, there is one extra pepper pixel.
    print(f"Noise {amount:.0%}: {number_salt} salt, {number_pepper} pepper pixels")
    return noisy


# Find edges using the Sobel and Canny methods from Assignment 1.
def detect_edges(image):
    """Reuse the Sobel and Canny steps from Assignment 1, Exercise 2(b)."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Fixed thresholds make the results before and after filtering comparable.
    _, sobel_edges = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
    canny_edges = cv2.Canny(image, 50, 150, L2gradient=True)
    return sobel_edges.astype(np.uint8), canny_edges


# Measure how much the filtered image differs from the original.
def mean_squared_error(original, filtered):
    difference = original.astype(float) - filtered.astype(float)
    return np.mean(difference ** 2)


# Apply dilation, erosion, opening and closing to the edge image.
def apply_morphology(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return {
        "unchanged": edges,
        "dilation": cv2.dilate(edges, kernel),
        "erosion": cv2.erode(edges, kernel),
        "opening": cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel),
        "closing": cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel),
    }


# Save an image and report an error if saving fails.
def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


# Arrange the results in a labelled grid for comparison.
def save_comparison(images, titles, path, columns):
    """Put equally sized images in a labelled grid for the report."""
    height, width = images[0].shape[:2]
    title_height = 45
    rows = (len(images) + columns - 1) // columns
    panel = np.full(
        (rows * (height + title_height), columns * width, 3), 255, dtype=np.uint8
    )

    for i, (image, title) in enumerate(zip(images, titles)):
        x = (i % columns) * width
        y = (i // columns) * (height + title_height)
        panel[y + title_height:y + title_height + height, x:x + width] = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        )
        cv2.putText(
            panel, title, (x + 6, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 0, 0), 1, cv2.LINE_AA
        )

    save_image(path, panel)


# Run the experiments for both noise levels and save the results.
def main():
    project_dir = Path(__file__).resolve().parent
    image_path = project_dir / "assignment-2-data" / "blox.jpg"
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    output_dir = project_dir / "outputs" / "exercise_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_sobel, original_canny = detect_edges(original)
    save_image(output_dir / "original.png", original)
    save_image(output_dir / "original_sobel.png", original_sobel)
    save_image(output_dir / "original_canny.png", original_canny)

    # Keep the random noise repeatable when comparing filter settings.
    np.random.seed(0)
    measurements = ["noise_amount,filter,setting,mse,selected"]

    for amount in (0.05, 0.1):
        noise_dir = output_dir / f"noise_{int(amount * 100):02d}"
        noise_dir.mkdir(exist_ok=True)
        noisy = add_salt_and_pepper_noise(original, amount)
        noisy_sobel, noisy_canny = detect_edges(noisy)
        save_image(noise_dir / "noisy.png", noisy)
        save_image(noise_dir / "noisy_sobel.png", noisy_sobel)
        save_image(noise_dir / "noisy_canny.png", noisy_canny)
        noisy_error = mean_squared_error(original, noisy)
        measurements.append(f"{amount},unfiltered,noisy,{noisy_error:.4f},False")

        # Each filter receives exactly the same noisy image at this noise level.
        # Gaussian sigma=0 lets OpenCV choose sigma from the kernel size.
        candidates = {
            "gaussian": {
                "3x3": cv2.GaussianBlur(noisy, (3, 3), 0),
                "5x5": cv2.GaussianBlur(noisy, (5, 5), 0),
                "7x7": cv2.GaussianBlur(noisy, (7, 7), 0),
            },
            "median": {
                "3x3": cv2.medianBlur(noisy, 3),
                "5x5": cv2.medianBlur(noisy, 5),
                "7x7": cv2.medianBlur(noisy, 7),
            },
            "bilateral": {
                "d5_c50_s5": cv2.bilateralFilter(noisy, 5, 50, 5),
                "d9_c100_s5": cv2.bilateralFilter(noisy, 9, 100, 5),
                "d9_c150_s5": cv2.bilateralFilter(noisy, 9, 150, 5),
            },
        }
        # Bilateral labels: d = diameter, c = sigmaColor, s = sigmaSpace.
        comparison_images = [original, original_sobel, original_canny,
                             noisy, noisy_sobel, noisy_canny]
        comparison_titles = ["Original", "Original Sobel", "Original Canny",
                             f"Noise {amount:.0%}", "Noisy Sobel", "Noisy Canny"]

        for filter_name, settings in candidates.items():
            filter_dir = noise_dir / filter_name
            filter_dir.mkdir(exist_ok=True)
            errors = {}
            images = []
            titles = []

            for setting, denoised in settings.items():
                sobel, canny = detect_edges(denoised)
                error = mean_squared_error(original, denoised)
                errors[setting] = error
                save_image(filter_dir / f"{setting}.png", denoised)
                save_image(filter_dir / f"{setting}_sobel.png", sobel)
                save_image(filter_dir / f"{setting}_canny.png", canny)
                images.extend([denoised, sobel, canny])
                titles.extend([f"{setting} MSE={error:.1f}", "Sobel", "Canny"])

            save_comparison(images, titles, filter_dir / "settings.png", columns=3)

            # The clean image is available, so use the lowest MSE to select a
            # denoised result. This does not necessarily give the best edge map.
            best_setting = min(errors, key=errors.get)
            best_image = settings[best_setting]
            best_sobel, best_canny = detect_edges(best_image)
            print(f"  {filter_name}: {best_setting}, MSE={errors[best_setting]:.2f}")
            for setting, error in errors.items():
                selected = setting == best_setting
                measurements.append(
                    f"{amount},{filter_name},{setting},{error:.4f},{selected}"
                )

            comparison_images.extend([best_image, best_sobel, best_canny])
            comparison_titles.extend([
                f"{filter_name}: {best_setting}", "Sobel", "Canny"
            ])

            # Apply morphology to the binary edges of the selected result.
            # Include erosion/opening even if they remove useful thin edges.
            morphology_images = []
            morphology_titles = []
            for edge_name, edges in [("Sobel", best_sobel), ("Canny", best_canny)]:
                for operation, result in apply_morphology(edges).items():
                    save_image(
                        filter_dir / f"best_{edge_name.lower()}_{operation}.png", result
                    )
                    morphology_images.append(result)
                    morphology_titles.append(f"{edge_name}: {operation}")
            save_comparison(
                morphology_images, morphology_titles,
                filter_dir / "morphology.png", columns=5
            )

        save_comparison(
            comparison_images, comparison_titles,
            noise_dir / "best_filters.png", columns=3
        )

    (output_dir / "filter_comparison.csv").write_text(
        "\n".join(measurements) + "\n", encoding="utf-8"
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
