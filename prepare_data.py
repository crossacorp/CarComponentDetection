"""
Unified Data Preparation Script
Combines: data collection + preprocessing + VLM dataset generation
Run this ONCE before training any models.
"""

import os
import time
import json
import random
import numpy as np
from scipy import ndimage
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import itertools
from datetime import datetime
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
CAR_MODEL_URL = "https://euphonious-concha-ab5c5d.netlify.app/"
OUTPUT_DIR = "dataset"
ANGLES_PER_STATE = 50  # 32 states × 50 angles = 1,600 images
RESUME_FROM_STATE = 1  # Set to state number to resume if interrupted
# ============================================================


class UnifiedDataPreparation:
    """Handles all data preparation: collection, preprocessing, and VLM dataset generation"""
    
    def __init__(self, url, output_dir):
        self.url = url
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        self.components = {
            'front_left': 'Front Left Door',
            'front_right': 'Front Right Door',
            'rear_left': 'Rear Left Door',
            'rear_right': 'Rear Right Door',
            'hood': 'Hood'
        }
        
        self.driver = None
        self.current_state = {k: 'closed' for k in self.components.keys()}
        self.current_angle = {'h': 0, 'v': 0}
    
    # ========================================
    # DATA COLLECTION
    # ========================================
    
    def setup_browser(self, headless=False):
        """Setup Chrome browser"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-gpu')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get(self.url)
        time.sleep(3)
    
    def reset_to_default_view(self):
        """Reset browser and car state"""
        self.driver.refresh()
        time.sleep(3)
        self.current_state = {k: 'closed' for k in self.current_state.keys()}
        self.current_angle = {'h': 0, 'v': 0}
    
    def rotate_camera_relative(self, delta_h, delta_v):
        """Rotate camera by relative amounts"""
        try:
            if delta_h == 0 and delta_v == 0:
                return
            
            canvas = self.driver.find_element(By.TAG_NAME, "canvas")
            canvas_width = canvas.size['width']
            canvas_height = canvas.size['height']
            safe_x = -int(canvas_width * 0.35)
            safe_y = -int(canvas_height * 0.35)
            
            h_pixels = int(delta_h * 0.8)
            v_pixels = int(delta_v * 0.8)
            
            MAX_MOVEMENT = 150
            h_pixels = max(-MAX_MOVEMENT, min(MAX_MOVEMENT, h_pixels))
            v_pixels = max(-MAX_MOVEMENT, min(MAX_MOVEMENT, v_pixels))
            
            action = ActionChains(self.driver)
            action.move_to_element_with_offset(canvas, safe_x, safe_y)
            action.click_and_hold()
            action.move_by_offset(h_pixels, v_pixels)
            action.release()
            action.perform()
            
            self.current_angle['h'] += delta_h
            self.current_angle['v'] += delta_v
            self.current_angle['h'] = self.current_angle['h'] % 360
            
            time.sleep(1.2)
            
        except Exception as e:
            print(f"    Warning: Could not rotate camera - {e}")
    
    def rotate_to_angle(self, target_h, target_v):
        """Rotate to absolute angle"""
        delta_h = target_h - self.current_angle['h']
        delta_v = target_v - self.current_angle['v']
        
        if delta_h > 180:
            delta_h -= 360
        elif delta_h < -180:
            delta_h += 360
        
        if abs(delta_h) > 150 or abs(delta_v) > 70:
            mid_h = self.current_angle['h'] + delta_h / 2
            mid_v = self.current_angle['v'] + delta_v / 2
            self.rotate_to_angle(mid_h, mid_v)
            self.rotate_to_angle(target_h, target_v)
            return
        
        if delta_h != 0 or delta_v != 0:
            self.rotate_camera_relative(delta_h, delta_v)
    
    def set_car_state(self, target_state):
        """Set car to target state using buttons"""
        button_mapping = {
            'front_left': 0,
            'front_right': 1,
            'rear_left': 2,
            'rear_right': 3,
            'hood': 4
        }
        
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        
        for component, desired_state in target_state.items():
            current = self.current_state.get(component, 'closed')
            if current != desired_state:
                button_index = button_mapping.get(component)
                if button_index is not None and button_index < len(buttons):
                    buttons[button_index].click()
                    self.current_state[component] = desired_state
                    time.sleep(2.5)
                    buttons = self.driver.find_elements(By.TAG_NAME, "button")
    
    def preprocess_screenshot(self, screenshot_bytes):
        """Preprocess screenshot: crop to car region, removing UI"""
        from io import BytesIO
        
        # Load image
        image = Image.open(BytesIO(screenshot_bytes)).convert('RGB')
        
        # Detect and crop to car
        bbox = self.detect_car_bbox(image)
        left, top, right, bottom = bbox
        cropped = image.crop((left, top, right, bottom))
        
        return cropped
    
    def detect_car_bbox(self, image, white_threshold=240, padding=10, ui_ignore_height=0.15):
        """Detect car bounding box while ignoring UI elements"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        non_white = gray < white_threshold
        
        # Ignore top portion (where UI is)
        ui_cutoff = int(image.height * ui_ignore_height)
        non_white[:ui_cutoff, :] = False
        
        # Find connected components
        labeled_array, num_features = ndimage.label(non_white)
        
        if num_features == 0:
            return 0, 0, image.width, image.height
        
        # Find largest component (the car)
        largest_size = 0
        largest_bbox = None
        
        for label_num in range(1, num_features + 1):
            component_mask = (labeled_array == label_num)
            component_size = np.sum(component_mask)
            
            if component_size > largest_size:
                largest_size = component_size
                
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
    
    def capture_and_save(self, state_dict, index, angle_info=""):
        """Capture screenshot, preprocess, and save"""
        time.sleep(1.0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"car_{index:05d}_{angle_info}_{timestamp}.png"
        
        # Capture screenshot
        screenshot_bytes = self.driver.get_screenshot_as_png()
        
        # Preprocess (crop to car)
        processed_image = self.preprocess_screenshot(screenshot_bytes)
        
        # Save processed image
        screenshot_path = os.path.join(self.images_dir, filename)
        processed_image.save(screenshot_path)
        
        # Save label
        label_data = {
            'filename': filename,
            'timestamp': timestamp,
            'components': state_dict,
            'index': index,
            'angle': angle_info,
            'camera_h': self.current_angle['h'],
            'camera_v': self.current_angle['v']
        }
        
        label_path = os.path.join(self.labels_dir, filename.replace('.png', '.json'))
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)
        
        return screenshot_path
    
    def collect_data(self, angles_per_state=50, resume_from_state=1):
        """Collect and preprocess data in one go"""
        print("="*80)
        print("STEP 1: DATA COLLECTION + PREPROCESSING")
        print("="*80)
        print(f"Random angles per state: {angles_per_state}")
        print(f"Images will be auto-cropped to remove UI")
        if resume_from_state > 1:
            print(f"Resuming from state {resume_from_state}")
        print("="*80)
        
        component_keys = list(self.components.keys())
        all_combinations = list(itertools.product([0, 1], repeat=5))
        
        total_images = len(all_combinations) * angles_per_state
        print(f"Total states: {len(all_combinations)}")
        print(f"Total images: {total_images}")
        print("="*80)
        
        existing_images = len([f for f in os.listdir(self.images_dir) if f.endswith('.png')]) if os.path.exists(self.images_dir) else 0
        index = existing_images
        print(f"Existing images: {existing_images}")
        print(f"Starting index: {index}")
        
        for combo_idx, combo in enumerate(all_combinations):
            if combo_idx < resume_from_state - 1:
                continue
            
            state_dict = {
                component_keys[i]: 'open' if combo[i] == 1 else 'closed'
                for i in range(5)
            }
            
            print(f"\n{'='*80}")
            print(f"State {combo_idx + 1}/{len(all_combinations)}: {state_dict}")
            print(f"{'='*80}")
            
            self.reset_to_default_view()
            self.set_car_state(state_dict)
            
            for angle_idx in range(angles_per_state):
                h_angle = random.uniform(0, 360)
                v_angle = random.uniform(-60, 60)
                
                angle_info = f"H{h_angle:06.1f}V{v_angle:+06.1f}".replace('.', 'p')
                print(f"  📷 Angle {angle_idx+1}/{angles_per_state}: h={h_angle:.1f}°, v={v_angle:.1f}°")
                
                self.rotate_to_angle(h_angle, v_angle)
                self.capture_and_save(state_dict, index, angle_info)
                index += 1
                time.sleep(0.2)
            
            print(f"  ✅ Completed state {combo_idx + 1}")
        
        print(f"\n{'=' * 80}")
        print(f"✅ Collection complete! Total images: {index}")
        print(f"{'=' * 80}")
        
        return index
    
    # ========================================
    # VLM DATASET GENERATION
    # ========================================
    
    def generate_vlm_description(self, components):
        """Generate varied natural language description"""
        open_doors = []
        closed_doors = []
        hood_open = components.get("hood", "closed") == "open"

        door_labels = {
            "front_left": "front left door",
            "front_right": "front right door",
            "rear_left": "rear left door",
            "rear_right": "rear right door",
        }

        for key, label in door_labels.items():
            if components.get(key, "closed") == "open":
                open_doors.append(label)
            else:
                closed_doors.append(label)

        def natural_join(items):
            if len(items) == 0:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return ", ".join(items[:-1]) + f", and {items[-1]}"

        # All closed
        if not open_doors and not hood_open:
            templates = [
                "All doors and the hood are closed. The car is fully shut.",
                "The car is completely closed. All four doors and the hood remain shut.",
                "Everything is closed — the front left door, front right door, rear left door, rear right door, and the hood are all shut.",
            ]
            return random.choice(templates)

        # All open
        if len(open_doors) == 4 and hood_open:
            templates = [
                "All doors and the hood are open. The car is fully open.",
                "Everything is open — all four doors and the hood are wide open.",
                "The car is completely open. The front left, front right, rear left, and rear right doors are all open, and so is the hood.",
            ]
            return random.choice(templates)

        # Mixed states
        templates = []

        if open_doors and closed_doors:
            open_str = natural_join(open_doors)
            closed_str = natural_join(closed_doors)

            if hood_open:
                templates.extend([
                    f"The {open_str} and the hood are open. The {closed_str} remain closed.",
                    f"The hood is open along with the {open_str}. The {closed_str} are closed.",
                ])
            else:
                templates.extend([
                    f"The {open_str} are open, while the {closed_str} and the hood remain closed.",
                    f"Only the {open_str} are open. The {closed_str} and the hood are closed.",
                ])

        if hood_open and not open_doors:
            templates.append("Only the hood is open. All four doors are closed.")

        if not templates:
            parts = []
            for key, label in door_labels.items():
                state = "open" if components.get(key) == "open" else "closed"
                parts.append(f"The {label} is {state}.")
            hood_state = "open" if hood_open else "closed"
            parts.append(f"The hood is {hood_state}.")
            return " ".join(parts)

        return random.choice(templates)
    
    def generate_vlm_dataset(self, val_split=0.1):
        """Generate VLM training dataset from collected images"""
        print("\n" + "="*80)
        print("STEP 2: VLM DATASET GENERATION")
        print("="*80)
        
        image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".png")])
        print(f"Found {len(image_files)} images")
        
        instruction = (
            "Look at this car image carefully. "
            "Describe the status of each component: "
            "front left door, front right door, rear left door, rear right door, and hood. "
            "State whether each one is open or closed."
        )
        
        dataset = []
        skipped = 0

        for img_file in tqdm(image_files, desc="Generating descriptions"):
            img_path = os.path.join(self.images_dir, img_file)
            label_path = os.path.join(self.labels_dir, img_file.replace(".png", ".json"))

            if not os.path.exists(label_path):
                skipped += 1
                continue

            with open(label_path, "r") as f:
                label_data = json.load(f)

            components = label_data.get("components", {})
            response_text = self.generate_vlm_description(components)

            dataset.append({
                "image_path": img_path,
                "instruction": instruction,
                "response": response_text,
                "components": components,
            })

        print(f"✅ Generated {len(dataset)} samples ({skipped} skipped)")

        # Shuffle and split
        random.seed(42)
        random.shuffle(dataset)

        val_size = max(1, int(len(dataset) * val_split))
        val_data = dataset[:val_size]
        train_data = dataset[val_size:]

        output = {
            "train": train_data,
            "val": val_data,
            "metadata": {
                "total": len(dataset),
                "train": len(train_data),
                "val": len(val_data),
                "instruction": instruction,
            }
        }

        output_file = "training_data.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n📁 Saved to: {output_file}")
        print(f"   Train: {len(train_data)} samples")
        print(f"   Val:   {len(val_data)} samples")
        
        return output_file
    
    def close(self):
        """Clean up"""
        if self.driver:
            self.driver.quit()


def main():
    print("="*80)
    print("UNIFIED DATA PREPARATION")
    print("Collects + Preprocesses + Generates VLM Dataset")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   States: 32 (all combinations)")
    print(f"   Angles per state: {ANGLES_PER_STATE}")
    print(f"   Total images: {32 * ANGLES_PER_STATE}")
    if RESUME_FROM_STATE > 1:
        remaining = 32 - RESUME_FROM_STATE + 1
        print(f"   Resuming from state: {RESUME_FROM_STATE}")
        print(f"   Remaining: {remaining * ANGLES_PER_STATE} images")
    print(f"   Output: {OUTPUT_DIR}/")
    print("="*80)
    
    prep = UnifiedDataPreparation(CAR_MODEL_URL, OUTPUT_DIR)
    
    try:
        print("\nSetting up browser...")
        prep.setup_browser(headless=False)
        
        input("\nPress Enter to start data preparation (or Ctrl+C to cancel)...")
        
        # Step 1: Collect and preprocess data
        total_images = prep.collect_data(
            angles_per_state=ANGLES_PER_STATE,
            resume_from_state=RESUME_FROM_STATE
        )
        
        # Step 2: Generate VLM dataset
        vlm_dataset_file = prep.generate_vlm_dataset()
        
        print("\n" + "="*80)
        print("🎉 SUCCESS! ALL DATA READY")
        print("="*80)
        print(f"✅ Collected & preprocessed: {total_images} images")
        print(f"✅ Generated VLM dataset: {vlm_dataset_file}")
        print(f"✅ Data saved to: {OUTPUT_DIR}/")
        print("\n📋 Next steps:")
        print("   1. Train CNN (Objective 1): python train.py")
        print("   2. Download VLM: python download_model.py")
        print("   3. Train VLM (Objective 2): python train_vlm.py")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nData preparation interrupted")
        print(f"\n💡 To resume, set RESUME_FROM_STATE to the last completed state + 1")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        prep.close()


if __name__ == "__main__":
    main()
