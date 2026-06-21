import os
import cv2
import numpy as np
from skimage.measure import perimeter, label, regionprops
from skimage.morphology import convex_hull_image


ROOT = "CBIS_DDSM_Kaggle"
outputFile = "features0512.csv"
columns = [
    "result","case","area","perimeter","circularity",
    "eccentricity","solidity","aspectRatio",
    "boundary_roughness","PAratio"
]

with open (outputFile, "w") as f:
    f.write(",".join(columns) + "\n")

for labelName in ["MALIGNANT", "BENIGN"]:
    classPath = os.path.join(ROOT, labelName)
    allFiles = len(os.listdir(classPath))
    for caseNum, caseName in enumerate(os.listdir(classPath)):
        print(labelName, caseNum + 1, "/", allFiles, caseName)
        casePath = os.path.join(classPath, caseName)
        imgPath = os.path.join(casePath, "full_image.png")
        maskPath = os.path.join(casePath, "roi_mask.png")

        # skip if image or mask doesn't exist
        if not os.path.exists(imgPath) or not os.path.exists(maskPath):
            continue

        # binary transformation
        mask = cv2.imread(maskPath, 0)
        maskBinary = (mask > 128).astype(np.uint8)

        # skip if size: full image != mask
        if maskBinary.shape != cv2.imread(imgPath).shape[:2]:
            print("Size mismatch, skipping:", caseName)
            continue

        # ===== Feature Calculation =====

        # Region properties
        labeled = label(maskBinary)
        regions = regionprops(labeled)
        region = max(regions, key = lambda r: r.area)  # select the largest region

        allArea = np.sum(maskBinary)
        assert region.area / allArea > 0.9, "The largest region should cover most of the area"

        # Area
        area = region.area

        # Perimeter
        peri = perimeter(region.image)

        # Circularity
        circularity = 0
        if peri > 0:
            circularity = 4 * np.pi * area / (peri ** 2)

        # Eccentricity
        eccentricity = region.eccentricity

        # Bounding box → Aspect ratio
        aspectRatio = region.minor_axis_length / region.major_axis_length if region.major_axis_length > 0 else 0

        # Convex hull
        convexMask = convex_hull_image(region.image)
        convexArea = np.sum(convexMask)
        convexPeri = perimeter(convexMask)

        # Solidity
        solidity = area / convexArea if convexArea > 0 else 0

        # Boundary roughness
        roughness = peri / convexPeri if convexPeri > 0 else 0

        # P/A ratio
        PAratio = peri / area if area > 0 else 0

        # ===== store the results =====
        results = [
            labelName,
            caseName,
            area,
            peri,
            circularity,
            eccentricity,
            solidity,
            aspectRatio,
            roughness,
            PAratio,
        ]

        with open (outputFile, "a") as f:
            f.write(",".join(map(str, results)) + "\n")
