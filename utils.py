"""
Shared utility functions for car component detection
"""

import numpy as np
from scipy import ndimage
from PIL import Image


def detect_car_bbox_ignore_ui(image, white_threshold=240, padding=10, ui_ignore_height=0.15):
    """
    Detect car bounding box while ignoring UI elements in top region.
    Uses connected component analysis to find largest region (the car).
    
    Args:
        image: PIL Image
        white_threshold: Threshold for considering pixels as white/background
        padding: Pixels to add around detected car region
        ui_ignore_height: Fraction of image height to ignore from top (where UI buttons are)
        
    Returns:
        tuple: (left, top, right, bottom) bounding box coordinates
    """
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Create binary mask: non-white pixels
    non_white = gray < white_threshold
    
    # Ignore top portion (where UI typically is)
    ui_cutoff = int(image.height * ui_ignore_height)
    non_white[:ui_cutoff, :] = False
    
    # Find connected components
    labeled_array, num_features = ndimage.label(non_white)
    
    if num_features == 0:
        # No car found, return full image
        return 0, 0, image.width, image.height
    
    # Find largest component (the car)
    largest_size = 0
    largest_bbox = None
    
    for label_num in range(1, num_features + 1):
        component_mask = (labeled_array == label_num)
        component_size = np.sum(component_mask)
        
        if component_size > largest_size:
            largest_size = component_size
            
            # Get bounding box of this component
            rows = np.any(component_mask, axis=1)
            cols = np.any(component_mask, axis=0)
            
            if rows.any() and cols.any():
                top, bottom = np.where(rows)[0][[0, -1]]
                left, right = np.where(cols)[0][[0, -1]]
                largest_bbox = (left, top, right, bottom)
    
    if largest_bbox is None:
        return 0, 0, image.width, image.height
    
    left, top, right, bottom = largest_bbox
    
    # Add padding
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    
    return left, top, right, bottom


def crop_to_car(image):
    """
    Crop full screenshot down to just the car region.
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: Cropped image containing only car
    """
    bbox = detect_car_bbox_ignore_ui(image)
    left, top, right, bottom = bbox
    cropped = image.crop((left, top, right, bottom))
    return cropped


def preprocess_image(image):
    """
    Preprocess image same as training data:
    1. Crop to car region (remove UI and whitespace)
    2. Return cropped image
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: Preprocessed image
    """
    return crop_to_car(image)
