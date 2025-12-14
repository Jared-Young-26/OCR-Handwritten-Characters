import cv2
import numpy as np
import scipy.ndimage as ndimage

def preprocess_canvas_to_mnist(canvas_img, mode="digit", input_type="canvas"):
    """
    Converts a raw canvas image or pre-cropped glyph into an MNIST/EMNIST-style
    28×28 normalized image suitable for classification.

    This function intentionally separates:
    - noise-heavy canvas input
    - already-cropped glyph input

    so that cleaning operations are not applied twice.
    """

    # --------------------------------------------------
    # 1. Convert input to grayscale
    #    (handles RGB, RGBA, and already-grayscale inputs)
    # --------------------------------------------------
    if len(canvas_img.shape) == 3:
        if canvas_img.shape[2] == 4:
            # Drop alpha channel if present
            canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = canvas_img.copy()

    # --------------------------------------------------
    # 2. Normalize foreground polarity
    #    Canvas drawings are black-on-white; MNIST expects white-on-black
    # --------------------------------------------------
    if np.mean(gray) > 127:
        gray = 255 - gray

    # --------------------------------------------------
    # 3. Ensure uint8 format for OpenCV operations
    # --------------------------------------------------
    if gray.dtype != np.uint8:
        gray = (gray * 255).clip(0, 255).astype(np.uint8)

    # --------------------------------------------------
    # 4. Foreground isolation
    #    Canvas input requires full cleanup; glyph input does not
    # --------------------------------------------------
    if input_type == "canvas":
        # Otsu threshold removes background and stabilizes stroke width
        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morphological opening removes small noise artifacts
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Compute bounding box around non-zero ink pixels
        coords = cv2.findNonZero(thresh)
        if coords is None:
            # Empty canvas → return blank MNIST image
            return np.zeros((28, 28), dtype=np.float32)

        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]

    else:
        # Glyph is already cropped — only binarize
        _, digit = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY
        )

    # --------------------------------------------------
    # 5. Aspect-ratio preserving resize
    #    MNIST digits occupy ~20×20 within a 28×28 canvas
    # --------------------------------------------------
    target = 20
    h_, w_ = digit.shape

    if h_ > w_:
        new_h = target
        new_w = int(w_ * target / h_)
    else:
        new_w = target
        new_h = int(h_ * target / w_)

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # --------------------------------------------------
    # 6. Zero-pad resized glyph into 28×28 canvas
    # --------------------------------------------------
    img28 = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    img28[y0:y0+new_h, x0:x0+new_w] = digit

    # --------------------------------------------------
    # 7. Normalize to [0, 1] float32
    # --------------------------------------------------
    img28 = img28.astype(np.float32) / 255.0

    # --------------------------------------------------
    # 8. EMNIST orientation correction
    #    EMNIST letters are rotated/flipped relative to MNIST digits
    # --------------------------------------------------
    if mode == "letter":
        img28 = np.rot90(img28, -1)
        img28 = np.fliplr(img28)

    # --------------------------------------------------
    # 9. Center glyph using center-of-mass
    #    This mimics MNIST dataset normalization
    # --------------------------------------------------
    cy, cx = ndimage.center_of_mass(img28)
    if not np.isnan(cx):
        shift_x = int(14 - cx)
        shift_y = int(14 - cy)
        img28 = np.roll(img28, (shift_y, shift_x), axis=(0, 1))

    # --------------------------------------------------
    # 10. Stroke normalization (letters only)
    #     Compensates for thinner EMNIST strokes
    # --------------------------------------------------
    if mode == "letter":
        kernel = np.ones((2, 2), np.uint8)
        img28 = cv2.dilate((img28 * 255).astype(np.uint8), kernel, 1)
        img28 = img28.astype(np.float32) / 255.0

    return img28



def compute_edges(image, threshold=0.001):
    """
    DESCRIPTION:
        Produces a binary edge map from a grayscale image using simplified
        Sobel-like gradient filters. This function exists to convert raw pixel
        intensity data into a structural representation—edges—which often carry
        more meaningful shape information for classification tasks.

    INPUT:
        image (np.ndarray):
            2D grayscale array of shape (H, W). Values may be in [0, 1] or [0, 255].
        threshold (float):
            Normalized cutoff in [0,1] determining what gradient strengths count as edges.

    PROCESSING:
        - Normalize the image to the range [0, 1] if necessary.
        - Apply horizontal and vertical Sobel filters to estimate intensity gradients.
        - Compute gradient magnitude at each pixel.
        - Normalize gradient magnitude so values fall in [0, 1].
        - Threshold the normalized gradient to produce a binary edge map.

    OUTPUT:
        edges (np.ndarray):
            Binary 2D array (H, W) of dtype float32, where:
                1.0 = edge pixel
                0.0 = non-edge pixel
    """
    # --- Normalize image to [0, 1] ---
    # If pixel values exceed 1.5, assume 0–255 and scale down.
    if image.max() > 1.5:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    # --- Define Sobel-like horizontal (Kx) and vertical (Ky) gradient filters ---
    # These filters approximate local intensity changes in x and y directions.
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    H, W = img.shape

    # Gradient containers for horizontal and vertical responses
    Gx = np.zeros_like(img)
    Gy = np.zeros_like(img)

    # --- Convolution: manually slide 3×3 Sobel kernels across the image ---
    # Ignore the 1-pixel border because edges cannot be computed there.
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            region = img[i-1:i+2, j-1:j+2]  # 3×3 region centered at (i, j)
            Gx[i, j] = np.sum(region * Kx)  # horizontal gradient strength
            Gy[i, j] = np.sum(region * Ky)  # vertical gradient strength

    # --- Gradient magnitude ---
    # Combines Gx and Gy into a single estimate of edge strength.
    mag = np.sqrt(Gx**2 + Gy**2)

    # Normalize gradient magnitude so thresholding is consistent across images
    mag = mag / (mag.max() + 1e-8)

    # --- Binary thresholding ---
    # Pixels with strong gradients are labeled as edges.
    edges = (mag > threshold).astype(np.float32)

    return edges


def segment_edge_image(edge_img, grid_rows=8, grid_cols=8):
    """
    DESCRIPTION:
        Converts a binary edge map into a structured feature vector by dividing
        the image into a uniform grid and measuring edge density in each cell.
        This provides the ANN with coarse structural information about where
        edges occur in the image.

    INPUT:
        edge_img (np.ndarray):
            2D binary array from compute_edges().
        grid_rows (int):
            Number of vertical segments.
        grid_cols (int):
            Number of horizontal segments.

    PROCESSING:
        - Divide the image into grid_rows × grid_cols segments.
        - For each segment, calculate edge density (mean of 0/1 values).
        - Flatten all segment densities into a 1D feature vector.

    OUTPUT:
        features (np.ndarray):
            1D float32 vector of length grid_rows * grid_cols.
            Each value is the edge density for one segment.
    """
    H, W = edge_img.shape
    seg_h = H // grid_rows
    seg_w = W // grid_cols

    features = []

    # --- Iterate across each grid cell ---
    for r in range(grid_rows):
        for c in range(grid_cols):

            # Compute segment boundaries (handle last row/col carefully)
            r_start = r * seg_h
            c_start = c * seg_w
            r_end = H if r == grid_rows - 1 else r_start + seg_h
            c_end = W if c == grid_cols - 1 else c_start + seg_w

            # Extract segment of the edge map
            segment = edge_img[r_start:r_end, c_start:c_end]

            # Compute density = proportion of pixels that are edges
            density = segment.mean()

            features.append(density)

    return np.array(features, dtype=np.float32)


def extract_edge_segment_features(image, grid_rows=8, grid_cols=8, threshold=0.001):
    """
    DESCRIPTION:
        Full pipeline step combining edge detection and grid segmentation.
        Converts an input image into a fixed-length structural feature vector.
        This function exists as the pre-processing stage for our custom ANN.

    INPUT:
        image (np.ndarray):
            2D grayscale image of shape (H, W).
        grid_rows, grid_cols (int):
            Specifies segmentation granularity.
        threshold (float):
            Edge threshold to use in compute_edges().

    PROCESSING:
        - Run edge detection using compute_edges().
        - Segment the resulting edge map with segment_edge_image().
        - Return the flattened density vector.

    OUTPUT:
        features (np.ndarray):
            1D float32 array of length grid_rows * grid_cols.
    """
    # Compute binary edge map
    edges = compute_edges(image, threshold=threshold)

    # Convert edge map into structural density features
    features = segment_edge_image(edges, grid_rows, grid_cols)

    return features

def segment_characters_from_word(img, min_area=20, dilate=True, return_boxes=False):
    """
    DESCRIPTION:
        Extracts individual handwritten characters from a word-level image.
        Performs thresholding, contour detection, bounding-box extraction,
        and spatial normalization to produce EMNIST-style 28×28 character patches.

    INPUT:
        img (np.ndarray):
            Input image array (grayscale, RGB, or RGBA).
        min_area (int):
            Minimum bounding-box area required to accept a contour as a character.
        dilate (bool):
            Whether to apply dilation to thicken strokes and improve
            contour consistency.
        return_boxes (bool):
            If True, returns both the processed character images and their bounding boxes.

    PROCESSING:
        - Convert input to grayscale.
        - Apply Otsu thresholding + inversion to isolate foreground ink.
        - Optionally dilate the ink to connect broken strokes.
        - Detect external contours and extract bounding rectangles.
        - Filter out small noise components below min_area.
        - Sort extracted boxes left-to-right for natural reading order.
        - For each box:
            * Crop the region of interest.
            * Embed into a square canvas to preserve aspect ratio.
            * Resize to 28×28.
            * Normalize pixel intensities to [0,1].

    OUTPUT:
        If return_boxes=False:
            [char28, char28, ...]
        If return_boxes=True:
            ([char28, ...], [(x, y, w, h), ...])
    """

    # --------------------------------------------------
    # Convert to grayscale (handles RGB and RGBA cases)
    # --------------------------------------------------
    if len(img.shape) == 3:
        # If alpha channel exists (BGRA), strip it
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # Convert BGR to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        gray = img.copy()

    # --------------------------------------------------
    # Threshold image:
    # - Otsu determines optimal threshold automatically.
    # - THRESH_BINARY_INV makes ink = white (255), background = black (0).
    # --------------------------------------------------
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------
    # Optional dilation to merge gaps in handwriting
    # and ensure contours are cleaner.
    # --------------------------------------------------
    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    # --------------------------------------------------
    # Extract outer contours — each contour ideally represents one character
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        thresh,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out very tiny specks (noise)
        if w * h < min_area:
            continue

        boxes.append((x, y, w, h))

    # --------------------------------------------------
    # Sort characters left to right so output aligns with reading order
    # --------------------------------------------------
    boxes.sort(key=lambda b: b[0])

    char_images = []

    # --------------------------------------------------
    # Convert each bounding box into a normalized 28×28 patch
    # --------------------------------------------------
    for (x, y, w, h) in boxes:
        # Crop the character region
        char_roi = thresh[y:y+h, x:x+w]

        # Use unified MNIST preprocessing
        #char28 = preprocess_canvas_to_mnist(char_roi)

        char_images.append(char_roi)


    # --------------------------------------------------
    # Return format depends on whether bounding boxes are requested
    # --------------------------------------------------
    if return_boxes:
        return char_images, boxes

    return char_images



def segment_words_from_line(img, min_area=20, dilate=True,
                            gap_factor=1.5, return_boxes=False):
    """
    DESCRIPTION:
        Segments a single handwritten line image into words and characters.
        Uses contour detection to find character blobs, then groups them into
        words based on horizontal gaps, and normalizes each character into a
        28×28 MNIST-style image.

    INPUT:
        img (np.ndarray):
            Input image array for one handwritten line (grayscale, RGB, or RGBA).
        min_area (int):
            Minimum bounding-box area required to treat a contour as a valid character.
        dilate (bool):
            Whether to apply dilation to strengthen and connect strokes.
        gap_factor (float):
            Multiplier applied to the median character width to determine what horizontal
            gap size counts as a word boundary.
        return_boxes (bool):
            If True, also returns the bounding boxes for all characters, grouped by word.

    PROCESSING:
        - Convert input to grayscale.
        - Apply Otsu thresholding with inversion to isolate foreground ink.
        - Optionally dilate to merge broken strokes.
        - Detect external contours and compute bounding boxes for candidate characters.
        - Filter out noise using a minimum area threshold.
        - Sort all boxes left-to-right.
        - Compute the median character width and derive a word-gap threshold.
        - Traverse boxes in order and start a new word whenever the horizontal gap
          between consecutive boxes exceeds the word-gap threshold.
        - For each character in each word:
            * Crop the bounding box region.
            * Embed it into a square canvas to preserve aspect ratio.
            * Resize to 28×28.
            * Normalize intensities to [0, 1].

    OUTPUT:
        If return_boxes=False:
            word_char_images : list of list of np.ndarray
                [
                  [char28_word1_char1, char28_word1_char2, ...],
                  [char28_word2_char1, ...],
                  ...
                ]

        If return_boxes=True:
            (word_char_images, word_char_boxes)
                word_char_images : as above
                word_char_boxes  : list of list of (x, y, w, h) bounding boxes
    """

    # --------------------------------------------------
    # Step 1: Normalize to grayscale (handles RGB / RGBA)
    # --------------------------------------------------
    if len(img.shape) == 3:
        # If the image has an alpha channel, drop it first
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # Convert BGR image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        gray = img.copy()

    # --------------------------------------------------
    # Step 2: Threshold + invert:
    # - Otsu chooses threshold automatically.
    # - Ink becomes white (255), background black (0).
    # --------------------------------------------------
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------
    # Step 3: Optional dilation to thicken strokes and
    # connect slightly broken character components.
    # --------------------------------------------------
    #if dilate:
    #    kernel = np.ones((3, 3), np.uint8)
    #    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # --------------------------------------------------
    # Step 4: Find external contours -> candidate characters
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        thresh,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip very small blobs that are likely noise
        if w * h < min_area:
            continue

        boxes.append((x, y, w, h))

    # If no valid characters found, return empty structures
    if not boxes:
        return ([], []) if return_boxes else []

    # --------------------------------------------------
    # Step 5: Sort characters left to right (reading order)
    # --------------------------------------------------
    boxes.sort(key=lambda b: b[0])

    # --------------------------------------------------
    # Step 6: Group boxes into words based on horizontal gaps
    # --------------------------------------------------
    # Estimate a "typical" character width
    widths = [w for (_, _, w, _) in boxes]
    median_w = np.median(widths)

    # Any gap larger than this is considered a word boundary
    word_gap_threshold = gap_factor * median_w

    words_boxes = []
    current = [boxes[0]]  # start first word with the leftmost box

    for box in boxes[1:]:
        x, y, w, h = box
        prev_x, prev_y, prev_w, prev_h = current[-1]

        # Horizontal gap between this box and the previous one
        gap = x - (prev_x + prev_w)

        if gap > word_gap_threshold:
            # Large gap -> start a new word
            words_boxes.append(current)
            current = [box]
        else:
            # Still part of the current word
            current.append(box)

    # Append the last accumulated word
    words_boxes.append(current)

    # --------------------------------------------------
    # Step 7: Build 28×28 normalized images for each char
    # in each word, and track bounding boxes if requested.
    # --------------------------------------------------
    word_char_images = []
    word_char_boxes = []

    for word in words_boxes:
        char_imgs_this_word = []
        boxes_this_word = []

        for (x, y, w, h) in word:
            char_roi = thresh[y:y+h, x:x+w]

            # Unified preprocessing
            #char28 = preprocess_canvas_to_mnist(char_roi)

            char_imgs_this_word.append(char_roi)
            boxes_this_word.append((x, y, w, h))

        word_char_images.append(char_imgs_this_word)
        word_char_boxes.append(boxes_this_word)

    # --------------------------------------------------
    # Step 8: Return images and (optionally) bounding boxes
    # --------------------------------------------------
    if return_boxes:
        return word_char_images, word_char_boxes

    return word_char_images

