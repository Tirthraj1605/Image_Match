from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os
import shutil

if __name__ == '__main__':
    fe = FeatureExtractor()
    images = Path('./static/uploaded').glob('*.jpeg')
    for image_path in images:
        # Construct new path
        new_path = Path('./static/img') / image_path.name
        # Move the image file
        shutil.move(image_path, new_path)
    for img_path in sorted(Path("./static/img").glob("*.jpeg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)