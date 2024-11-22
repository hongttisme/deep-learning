import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import random
from matplotlib import pyplot as plt
import io

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


uploaded_image_path = None
scale_value = None
show_category_var = None
show_confidence_var = None
img_label = None

def main_menu():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(
        root,
        text="Please select what you want to do",
        font=("Comic Sans MS", 25)
    ).pack(pady=50)

    tk.Button(
        root,
        text="Image",
        font=("Comic Sans MS", 15),
        width=15,
        height=2,
        relief="groove",
        command=image_ui
    ).pack(pady=20)

    tk.Button(
        root,
        text="Real Time",
        font=("Comic Sans MS", 15),
        width=15,
        height=2,
        relief="groove",
        command=real_time_ui
    ).pack(pady=20)

def image_ui():
    global img_label, uploaded_image_path, show_category_var, show_confidence_var, scale_value

    uploaded_image_path = None

    for widget in root.winfo_children():
        widget.destroy()

    left_frame = tk.Frame(root, width=250, bg="lightgray", relief="ridge", bd=2)
    left_frame.pack_propagate(False)
    left_frame.pack(side="left", fill="y")

    button_list = [
        ("Upload Image", upload_image),
        ("Save as Annotation", save_as_annotation),
        ("Main Menu", main_menu)
    ]

    for text, command in button_list:
        tk.Button(
            left_frame,
            text=text,
            font=("Comic Sans MS", 12),
            width=15,
            height=2,
            relief="groove",
            command=command
        ).pack(pady=75)

    center_frame = tk.Frame(root, bg="white", relief="ridge", bd=2)
    center_frame.pack(side="left", fill="both", expand=True)

    img_label = tk.Label(center_frame, bg="white")
    img_label.pack(expand=True)

    right_frame = tk.Frame(root, width=250, bg="lightgray", relief="ridge", bd=2)
    right_frame.pack_propagate(False)
    right_frame.pack(side="right", fill="y")

    scale_value = tk.DoubleVar(value=0.25)

    tk.Label(
        right_frame,
        text="Confidence",
        font=("Comic Sans MS", 12),
        bg="lightgray"
    ).pack(pady=10)

    scale = tk.Scale(
        right_frame,
        from_=0,
        to=1,
        resolution=0.01,
        orient="horizontal",
        length=200,
        variable=scale_value,
        bg="lightgray"
    )
    scale.pack(pady=10)

    show_category_var = tk.IntVar(value=1)
    tk.Checkbutton(
        right_frame,
        text="Show Category",
        variable=show_category_var,
        font=("Comic Sans MS", 12),
        bg="lightgray"
    ).pack(pady=20)

    show_confidence_var = tk.IntVar(value=1)
    tk.Checkbutton(
        right_frame,
        text="Show Confidence",
        variable=show_confidence_var,
        font=("Comic Sans MS", 12),
        bg="lightgray"
    ).pack(pady=20)

    tk.Button(
        right_frame,
        text="Detect",
        font=("Comic Sans MS", 15),
        width=15,
        height=2,
        relief="groove",
        command=detect_image
    ).pack(pady=20)

def upload_image():
    global img_label, uploaded_image_path

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    
    if file_path:
        uploaded_image_path = file_path
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        img_label.config(image=img_tk)
        img_label.image = img_tk

def detect_image():
    global img_label, uploaded_image_path, scale_value

    color = (random.random(),random.random(),random.random())
    if not uploaded_image_path:
        messagebox.showwarning("Warning", "No image uploaded.")
        return

    model.conf = scale_value.get()
    results = model(uploaded_image_path)
    numpy_results = results.xyxy[0].cpu().numpy()

    img = cv2.imread(uploaded_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    for det in numpy_results:
        xmin, ymin, xmax, ymax, confidence, cls = det[:6]
        label = ""

        if show_category_var.get():
            label += results.names[int(cls)]

        if show_confidence_var.get():
            label += f" {round(confidence*100,2)}%"

        ax.add_patch(plt.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), color=color, fill=False, linewidth=2))
        ax.text(xmin + 2, ymin - 5, label, color='white', fontsize=12, backgroundcolor=color)

    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    detected_image = Image.open(buf)
    img_tk = ImageTk.PhotoImage(detected_image)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    buf.close()

def save_as_annotation():
    if not uploaded_image_path:
        messagebox.showwarning("Warning", "No image uploaded.")
        return

    results = model(uploaded_image_path)
    df = results.pandas().xyxy[0]
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if save_path:
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Info", "Annotations saved successfully!")

def real_time_ui():
    pass

root = tk.Tk()
root.geometry("1200x900")
main_menu()

root.mainloop()
