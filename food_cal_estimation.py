import cv2
import pandas as pd
import pyttsx3
import threading
import time
import os
import queue
import tkinter as tk
from tkinter import ttk
import numpy as np
from collections import defaultdict, deque
from scipy.interpolate import interp1d
from inference_sdk import InferenceHTTPClient

# ========== Text-to-Speech Setup ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        print("🗣️", text)
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# ========== Utensil Dropdown Selection ==========
def get_utensil_choice():
    selected = [None]
    def select():
        utensil = combo.get().lower()
        selected[0] = utensil
        root.destroy()
    root = tk.Tk()
    root.title("Select Utensil Type")
    ttk.Label(root, text="Choose Utensil (Top View):").pack(pady=10)
    combo = ttk.Combobox(root, values=["Bowl", "Plate", "Glass"], state="readonly")
    combo.pack()
    combo.set("Bowl")
    ttk.Button(root, text="Start", command=select).pack(pady=10)
    root.mainloop()
    return selected[0]

utensil_type = get_utensil_choice()

# ========== Utensil Data ==========
UTENSIL_DATA = {
    "bowl": [
        {"name": "Katori (Small)", "min_diameter": 7, "max_diameter": 9, "min_height": 4, "max_height": 5, "min_volume": 100, "max_volume": 200},
        {"name": "Rice Bowl", "min_diameter": 12, "max_diameter": 15, "min_height": 6, "max_height": 8, "min_volume": 400, "max_volume": 600},
        {"name": "Soup Bowl", "min_diameter": 10, "max_diameter": 12, "min_height": 6, "max_height": 8, "min_volume": 250, "max_volume": 400}
    ],
    "plate": [
        {"name": "Quarter Plate", "min_diameter": 15, "max_diameter": 18, "min_height": 2, "max_height": 4, "min_volume": 350, "max_volume": 700},
        {"name": "Dinner Plate (Thali)", "min_diameter": 25, "max_diameter": 28, "min_height": 4, "max_height": 6, "min_volume": 1400, "max_volume": 2000}
    ],
    "glass": [
        {"name": "Water Glass", "min_diameter": 6, "max_diameter": 8, "min_height": 10, "max_height": 12, "min_volume": 200, "max_volume": 300}
    ]
}

def classify_utensil(diameter_cm, utensil_type):
    for utensil in UTENSIL_DATA.get(utensil_type, []):
        if utensil["min_diameter"] <= diameter_cm <= utensil["max_diameter"]:
            return utensil
    return None

def estimate_height(diameter_cm, utensil_type):
    utensils = UTENSIL_DATA.get(utensil_type, [])
    diameters = [u["min_diameter"] for u in utensils]
    heights = [u["min_height"] for u in utensils]
    if not diameters or not heights:
        return 5  # default height
    interp_func = interp1d(diameters, heights, kind='linear', fill_value="extrapolate")
    return float(interp_func(diameter_cm))

def estimate_volume_cm(diameter_cm, height_cm, utensil_type):
    utensils = UTENSIL_DATA.get(utensil_type, [])
    diameters = [u["min_diameter"] for u in utensils]
    volumes = [u["min_volume"] for u in utensils]
    if not diameters or not volumes:
        return 0
    interp_func = interp1d(diameters, volumes, kind='linear', fill_value="extrapolate")
    return float(interp_func(diameter_cm))

# ========== Nutrition File ==========
csv_path = "Indian_Food_Nutrition_Processed - Copy (2).csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

nutrition_df = pd.read_csv(csv_path)
nutrition_df.columns = nutrition_df.columns.str.strip().str.lower()
nutrition_info = {
    str(row['dish name']).strip().lower(): {
        "calories": row.get("calories (kcal)", 0),
        "protein": row.get("protein (g)", 0),
        "fat": row.get("fats (g)", 0),
        "carbs": row.get("carbohydrates (g)", 0)
    }
    for _, row in nutrition_df.iterrows()
}

# ========== Roboflow Client Setup ==========
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="J5WFqAyDJ4oQRVxjeLL9"
)

# ========== Camera Setup ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open camera")

COIN_DIAMETER_CM = 2.3
pixels_per_cm = None

bowl_r_px_history = deque(maxlen=20)

summary_totals = defaultdict(float)
detection_results = []

accumulated_foods = []

def estimate_volume(w_px, h_px, label=""):
    if pixels_per_cm is None:
        return 0
    diameter_cm = w_px / pixels_per_cm
    classified = classify_utensil(diameter_cm, utensil_type)
    if classified:
        height_cm = classified["min_height"]
    else:
        height_cm = estimate_height(diameter_cm, utensil_type)
    volume_ml = estimate_volume_cm(diameter_cm, height_cm, utensil_type)
    if label in ["plain rice", "biryani", "poha", "upma", "halwa"]:
        volume_ml *= 1.15
    return volume_ml

def save_to_csv(data_list, filename="detected_meals.csv"):
    df = pd.DataFrame(data_list)
    df.to_csv(filename, index=False)
    print(f"CSV updated with {len(data_list)} entries.")

def run_inference(frame_copy):
    global detection_results, summary_totals, accumulated_foods
    cv2.imwrite("frame.jpg", frame_copy)
    try:
        result = CLIENT.infer("frame.jpg", model_id="rice-detection-1a7vq/2")
    except Exception as e:
        print("❌ Inference error:", e)
        return

    detection_results.clear()
    summary_totals = defaultdict(float)
    summary = ""

    stable_bowl = False
    avg_radius_px = 0
    std_dev = 0

    if len(bowl_r_px_history) == bowl_r_px_history.maxlen:
        std_dev = np.std(bowl_r_px_history)
        if std_dev < 1.5:
            stable_bowl = True
            avg_radius_px = np.mean(bowl_r_px_history)
        else:
            median = np.median(bowl_r_px_history)
            clipped = np.clip(bowl_r_px_history, median - 5, median + 5)
            avg_radius_px = np.mean(clipped)
    elif len(bowl_r_px_history) > 0:
        avg_radius_px = np.mean(bowl_r_px_history)

    bowl_volume_ml = 0
    if stable_bowl and pixels_per_cm is not None:
        bowl_diameter_cm = (2 * avg_radius_px) / pixels_per_cm
        classified_bowl = classify_utensil(bowl_diameter_cm, utensil_type)
        if classified_bowl:
            bowl_height_cm = classified_bowl["min_height"]
        else:
            bowl_height_cm = estimate_height(bowl_diameter_cm, utensil_type)
        bowl_volume_ml = estimate_volume_cm(bowl_diameter_cm, bowl_height_cm, utensil_type)

    for pred in result.get('predictions', []):
        label = pred['class'].strip().lower()
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        volume_ml = estimate_volume(w, h, label)
        nut = nutrition_info.get(label)
        if nut:
            factor = volume_ml / 100.0
            for key in ['calories', 'protein', 'fat', 'carbs']:
                summary_totals[key] += round(nut[key] * factor, 1)
            summary += f"{label.title()} ≈ {int(volume_ml)}ml → {round(nut['calories'] * factor)} kcal\n"

            detected_item = {
                "label": label,
                "volume_ml": round(volume_ml, 1),
                "calories": round(nut["calories"] * factor, 1),
                "protein": round(nut["protein"] * factor, 1),
                "fat": round(nut["fat"] * factor, 1),
                "carbs": round(nut["carbs"] * factor, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            accumulated_foods.append(detected_item)

        food_ratio = 0
        if bowl_volume_ml > 0:
            food_ratio = volume_ml / bowl_volume_ml

        detection_results.append({
            "label": label, "x": x, "y": y, "width": w, "height": h,
            "volume_ml": volume_ml, "nutrition": nut, "food_ratio": food_ratio
        })

    total_summary = f"Total: {int(summary_totals['calories'])} kcal, {summary_totals['protein']}g protein, {summary_totals['fat']}g fat, {summary_totals['carbs']}g carbs"

    if stable_bowl and bowl_volume_ml > 0:
        summary += f"Bowl Volume ≈ {int(bowl_volume_ml)} ml\n"

    if summary:
        if bowl_volume_ml > 0:
            avg_food_ratio = np.mean([d["food_ratio"] for d in detection_results]) if detection_results else 0
            ratio_percent = avg_food_ratio * 100
            ratio_text = f"\nFood fills approx {ratio_percent:.1f}% of the bowl."
            speak(summary.strip() + ratio_text + "\n" + total_summary)
        else:
            speak(summary.strip() + "\n" + total_summary)

    save_to_csv(accumulated_foods)

print("📷 Starting real-time nutrition detection...")

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.4
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 1
LINE_TYPE = cv2.LINE_AA
BG_COLOR = (0, 0, 0)

def put_retro_text(img, text, pos, bg_color=BG_COLOR, font=FONT, scale=FONT_SCALE, color=FONT_COLOR, thickness=FONT_THICKNESS):
    x, y = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 2, y - h - 2), (x + w + 2, y + 2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, LINE_TYPE)

def main():
    global pixels_per_cm
    last_infer_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_copy = frame.copy()

        coin_circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                        param1=50, param2=30, minRadius=10, maxRadius=50)
        if coin_circles is not None:
            coin_circles = np.uint16(np.around(coin_circles))
            coin_r_px = coin_circles[0][0][2]
            pixels_per_cm = (coin_r_px * 2) / COIN_DIAMETER_CM
            cv2.circle(frame_copy, (coin_circles[0][0][0], coin_circles[0][0][1]), coin_r_px, (255, 0, 0), 2)
            put_retro_text(frame_copy, f"Coin radius px: {coin_r_px}", (10, 20))
        else:
            pixels_per_cm = None
            put_retro_text(frame_copy, "Coin not detected", (10, 20), bg_color=(0,0,255))

        bowl_circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                                        param1=50, param2=40, minRadius=40, maxRadius=150)
        if bowl_circles is not None:
            bowl_circles = np.uint16(np.around(bowl_circles))
            bowl_r_px = bowl_circles[0][0][2]
            bowl_r_px_history.append(bowl_r_px)
            cv2.circle(frame_copy, (bowl_circles[0][0][0], bowl_circles[0][0][1]), bowl_r_px, (0, 255, 255), 2)
            put_retro_text(frame_copy, f"Bowl radius px: {bowl_r_px}", (10, 40))
        else:
            put_retro_text(frame_copy, "Bowl not detected", (10, 40), bg_color=(0,0,255))

        current_time = time.time()
        if current_time - last_infer_time > 3:
            threading.Thread(target=run_inference, args=(frame_copy.copy(),), daemon=True).start()
            last_infer_time = current_time

        y_offset = 60
        for det in detection_results:
            x, y, w, h = det["x"], det["y"], det["width"], det["height"]
            label = det["label"].title()
            volume_ml = det["volume_ml"]
            nut = det["nutrition"]

            # Correct bounding box drawing with center coordinates:
            top_left = (x - w // 2, y - h // 2)
            bottom_right = (x + w // 2, y + h // 2)
            cv2.rectangle(frame_copy, top_left, bottom_right, (0, 255, 0), 1)
            put_retro_text(frame_copy, f"{label}", (top_left[0], top_left[1] - 5))
            if nut:
                text = f"Vol: {int(volume_ml)}ml Cal: {int(nut['calories'])} Prot: {nut['protein']}g Fat: {nut['fat']}g Carb: {nut['carbs']}g"
                put_retro_text(frame_copy, text, (top_left[0], bottom_right[1] + 15))

            summary_text = (
                f"Calories: {int(summary_totals['calories'])} kcal  "
                f"Protein: {summary_totals['protein']:.1f} g  "
                f"Fat: {summary_totals['fat']:.1f} g  "
                f"Carbs: {summary_totals['carbs']:.1f} g"
            )
            put_retro_text(frame_copy, summary_text, (10, frame_copy.shape[0] - 20))

        cv2.imshow("Nutrition Detector", frame_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)  # stop speech thread

if __name__ == "__main__":
    main()
