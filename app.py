import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, Text
from PIL import Image, ImageTk
import pytesseract
import requests
import cv2
import numpy as np
import re
import os

output_base_dir = os.path.join(os.path.dirname(__file__), "output")

# Levels for OCR
class Levels:
    PAGE = 1
    BLOCK = 2
    PARAGRAPH = 3
    LINE = 4
    WORD = 5

# Helper functions
def create_directories(img_name, column_no=0):
    img_dir = os.path.join(output_base_dir, img_name)
    columns_dir = os.path.join(img_dir, "columns/column_" + str(column_no))
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(columns_dir, exist_ok=True)
    return img_dir, columns_dir

def threshold_image(img_src, img_name, output_dir):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_gray.png"), img_gray)
    
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_thresh.png"), img_thresh)
    
    return img_thresh, img_gray

def mask_image(img_src, lower, upper, img_name, output_dir):
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_hsv.png"), img_hsv)
    
    hsv_lower = np.array(lower, np.uint8)
    hsv_upper = np.array(upper, np.uint8)
    img_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_mask.png"), img_mask)
    
    return img_mask, img_hsv

def denoise_image(img_src, img_name, output_dir):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img_denoise = cv2.morphologyEx(img_src, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_denoise.png"), img_denoise)
    
    return img_denoise

def find_highlighted_words(img_mask, data_ocr, img_name, output_dir, threshold_percentage=25):
    data_ocr['highlighted'] = [False] * len(data_ocr['text'])
    for i in range(len(data_ocr['text'])):
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top'][i], data_ocr['width'][i], data_ocr['height'][i])
        rect_threshold = (w * h * threshold_percentage) / 100
        img_roi = img_mask[y:y+h, x:x+w]
        count = cv2.countNonZero(img_roi)
        if count > rect_threshold:
            data_ocr['highlighted'][i] = True
            
    # Create an image with highlighted words
    img_highlighted = img_mask.copy()
    for i in range(len(data_ocr['text'])):
        if data_ocr['highlighted'][i]:
            (x, y, w, h) = (data_ocr['left'][i], data_ocr['top'][i], data_ocr['width'][i], data_ocr['height'][i])
            cv2.rectangle(img_highlighted, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_highlighted.png"), img_highlighted)
    
    return data_ocr

def words_to_string(data_ocr):
    word_list = []
    line_breaks = (Levels.PAGE, Levels.BLOCK, Levels.PARAGRAPH, Levels.LINE)
    for i in range(len(data_ocr['text'])):
        if data_ocr['level'][i] in line_breaks:
            word_list.append("\n")
            continue
        text = data_ocr['text'][i].strip()
        if text and data_ocr['highlighted'][i]:
            word_list.append(text + " ")
    word_string = "".join(word_list)
    word_string = re.sub(r'\n+', '\n', word_string).strip()
    return word_string

def load_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return
    image = Image.open(filepath)
    image.thumbnail((400, 400))
    img_display    = ImageTk.PhotoImage(image)
    img_label.config(image=img_display)
    img_label.image = img_display
    img_label.filepath = filepath
    highlighted_text, summary = process_image(filepath)
    display_text(highlighted_text, summary)

def detect_columns(img, img_name, output_dir):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_bin.png"), img_bin)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 10))
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=1)  # expansion + erosion
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    column_bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > img_bin.size / 10]
    column_bounding_boxes = sorted(column_bounding_boxes, key=lambda b: b[0])
    
    img_bin_colored = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)  # Convert binary image to color image for drawing
    for (x, y, w, h) in column_bounding_boxes:
        cv2.rectangle(img_bin_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color
    
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_columns.png"), img_bin_colored)
    
    return column_bounding_boxes

def process_image(filepath):
    img_name = os.path.splitext(os.path.basename(filepath))[0]
    img_dir, columns_dir = create_directories(img_name)
    
    img = cv2.imread(filepath)
    cv2.imwrite(os.path.join(img_dir, f"{img_name}_original.png"), img)
    
    column_bounding_boxes = detect_columns(img, img_name, img_dir)
    
    all_highlighted_text = []
    
    for idx, (x, y, w, h) in enumerate(column_bounding_boxes):
        mg_dir, columns_dir = create_directories(img_name, idx)
        column = img[y:y+h, x:x+w]
        img_thresh, _ = threshold_image(column, f"{img_name}_column_{idx}", columns_dir)
        data_ocr = pytesseract.image_to_data(img_thresh, lang='eng', config='--psm 6', output_type=pytesseract.Output.DICT)
        hsv_lower = [22, 30, 30]
        hsv_upper = [45, 255, 255]
        img_mask, _ = mask_image(column, hsv_lower, hsv_upper, f"{img_name}_column_{idx}", columns_dir)
        img_mask_denoised = denoise_image(img_mask, f"{img_name}_column_{idx}", columns_dir)
        data_ocr = find_highlighted_words(img_mask_denoised, data_ocr, f"{img_name}_column_{idx}", columns_dir, threshold_percentage=25)
        highlighted_text = words_to_string(data_ocr)
        all_highlighted_text.append(highlighted_text)
    
    full_text = "\n".join(all_highlighted_text)
    summary = summarize_text(full_text)
    return full_text, summary

def summarize_text(text):
    url = 'http://localhost:8000/summ'  # Adjust the URL to your actual summarization API endpoint
    response = requests.post(url, json={'text': text})
    if response.status_code == 200:
        return response.json().get('summary')
    else:
        return "Failed to summarize text"

def display_text(highlighted_text, summary):
    extracted_text_display.config(state=tk.NORMAL)
    summary_display.config(state=tk.NORMAL)
    extracted_text_display.delete(1.0, tk.END)
    summary_display.delete(1.0, tk.END)
    extracted_text_display.insert(tk.END, highlighted_text)
    summary_display.insert(tk.END, summary)
    extracted_text_display.config(state=tk.DISABLED)
    summary_display.config(state=tk.DISABLED)

# GUI Setup
root = tk.Tk()
root.title("Image Text Summarizer")
root.minsize(600, 400)
root.configure(bg="white")  # Set background of window to white

# Create a Style object
style = ttk.Style()

# Configure a new style for the TButton element
style.configure('Default.TButton', foreground='white', font=('Arial', 12))

# Configure grid layout to be responsive
root.grid_rowconfigure(3, weight=1)  # Adjusted to accommodate labels
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Font configuration for better visibility
font_large = ('Arial', 12)  # Larger font for better readability

# Widgets
upload_button = ttk.Button(root, text="Upload Image", style='Default.TButton', command=load_image)
upload_button.grid(row=0, column=0, columnspan=2, pady=10)

img_label = tk.Label(root, bg="white")  # Background set to white
img_label.grid(row=1, column=0, columnspan=2, pady=10)

# Labels for text displays
extracted_text_label = tk.Label(root, text="Extracted Text", bg="white", fg="black", font=font_large)
extracted_text_label.grid(row=2, column=0, padx=(10, 5), pady=(10, 0), sticky="nw")

summary_label = tk.Label(root, text="Summary", bg="white", fg="black", font=font_large)
summary_label.grid(row=2, column=1, padx=(5, 10), pady=(10, 0), sticky="nw")

# Text displays with enhanced contrast and larger font
extracted_text_display = Text(root, height=10, wrap="word", bg="#f0f0f0", fg="black", relief="solid", borderwidth=1, font=font_large, highlightbackground="#CCCCCC", highlightthickness=1)  # Slightly off-white background for text area
extracted_text_display.grid(row=3, column=0, padx=(10, 5), pady=(5, 10), sticky="nsew")

summary_display = Text(root, height=10, wrap="word", bg="#f0f0f0", fg="black", relief="solid", borderwidth=1, font=font_large, highlightbackground="#CCCCCC", highlightthickness=1)  # Slightly off-white background for text area
summary_display.grid(row=3, column=1, padx=(5, 10), pady=(5, 10), sticky="nsew")

root.mainloop()
