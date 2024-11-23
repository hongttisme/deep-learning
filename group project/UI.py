import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import random
from matplotlib import pyplot as plt
import io

# import model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Global Variable
uploaded_image_path = None
scale_value = None
show_category_var = None
show_confidence_var = None
img_label = None
cap = None
running = False

# Main Menu Function
def main_menu():
    
    #clear all current page all things
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
    #the global variable we will use
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

    #create 3 button
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

    #Open the file then limited only image file format
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    
    #read img -> convert rgb format -> Numpy format to PIL format -> From PIL convert to Tkinter format -> config is show the image 
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

    #Using scale_value to control the model confidence threshold
    model.conf = scale_value.get()
    results = model(uploaded_image_path)
    
    # Convert the result to numpy style
    numpy_results = results.xyxy[0].cpu().numpy()

    #read the image convert image to RGB
    img = cv2.imread(uploaded_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # since the tkinter cannot directly draw the bbox, so we secretly painted it ourselves inside
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    #draw the result
    for det in numpy_results:
        xmin, ymin, xmax, ymax, confidence, cls = det[:6]
        label = ""

        if show_category_var.get():
            label += results.names[int(cls)]

        if show_confidence_var.get():
            label += f" {confidence:.2f}"

        ax.add_patch(plt.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), color=color, fill=False, linewidth=2))
        ax.text(xmin + 2, ymin - 5, label, color='white', fontsize=12, backgroundcolor=color)

    ax.axis("off")
    
    #Save the image, after drawing figure will not store into physical memory location
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    #show after drawing bbox figure
    detected_image = Image.open(buf)
    img_tk = ImageTk.PhotoImage(detected_image)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    buf.close()

def save_as_annotation():

    if not uploaded_image_path:
        messagebox.showwarning("Warning", "No image uploaded.")
        return

    #directly take the data and write the data as csv format in text file
    results = model(uploaded_image_path)
    df = results.pandas().xyxy[0]
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if save_path:
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Info", "Annotations saved successfully!")

def real_time_ui():
    global cap, running

    for widget in root.winfo_children():
        widget.destroy()

    #Open the camera
    running = True
    cap = cv2.VideoCapture(0)

    #create an area to show the video
    video_label = tk.Label(root)
    video_label.pack(expand=True, fill="both")

    def update_video():
        
        #if the video is running then detect each frame then draw the bbox
        if running:
            ret, frame = cap.read()
            if ret:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                detections = results.xyxy[0].cpu().numpy()

                for det in detections:
                    xmin, ymin, xmax, ymax, conf, cls = det[:6]
                    label = f"{results.names[int(cls)]} {conf:.2f}"
                
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                #each frame will show in the User Interface
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                video_label.config(image=img)
                video_label.image = img
            #looping to update the video
            root.after(1, update_video)

    def stop_video():
        global running, cap
        running = False
        if cap.isOpened():
            cap.release()
        main_menu()

    back_button = tk.Button(root, text="Main Menu", command=stop_video, font=("Comic Sans MS", 15))
    back_button.pack(pady=20)

    update_video()

#initialize the UI
root = tk.Tk()
root.geometry("1200x900")
main_menu()

root.mainloop()
