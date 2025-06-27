import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import os
import sys
from PIL import Image, ImageTk
import cv2

class FaceSwapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Face Swap - GUI")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        
        # Variables
        self.source_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.face_attr_direction = tk.StringVar()
        self.resolution = tk.IntVar(value=128)
        self.face_attr_steps = tk.DoubleVar(value=0.0)
        self.delay = tk.IntVar(value=0)
        self.obs_enabled = tk.BooleanVar()
        self.mouth_mask = tk.BooleanVar()
        self.fps_delay = tk.BooleanVar()
        
        # Process tracking
        self.process = None
        self.is_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Live Face Swap Configuration", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Required Settings Frame
        req_frame = ttk.LabelFrame(main_frame, text="Required Settings", padding=10)
        req_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Source Image
        ttk.Label(req_frame, text="Source Face Image:").pack(anchor=tk.W)
        source_frame = ttk.Frame(req_frame)
        source_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(source_frame, textvariable=self.source_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(source_frame, text="Browse", command=self.browse_source).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Model Path
        ttk.Label(req_frame, text="Model Path (.pth file):").pack(anchor=tk.W)
        model_frame = ttk.Frame(req_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Basic Settings Frame
        basic_frame = ttk.LabelFrame(main_frame, text="Basic Settings", padding=10)
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Resolution
        res_frame = ttk.Frame(basic_frame)
        res_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(res_frame, text="Resolution:").pack(side=tk.LEFT)
        res_spinbox = ttk.Spinbox(res_frame, from_=64, to=512, textvariable=self.resolution, width=10)
        res_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(res_frame, text="pixels (face crop size)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Delay
        delay_frame = ttk.Frame(basic_frame)
        delay_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(delay_frame, text="Delay:").pack(side=tk.LEFT)
        delay_spinbox = ttk.Spinbox(delay_frame, from_=0, to=5000, increment=50, textvariable=self.delay, width=10)
        delay_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(delay_frame, text="milliseconds").pack(side=tk.LEFT, padx=(5, 0))
        
        # Advanced Settings Frame
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Settings", padding=10)
        adv_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Face Attribute Direction
        ttk.Label(adv_frame, text="Face Attribute Direction (.npy file):").pack(anchor=tk.W)
        attr_frame = ttk.Frame(adv_frame)
        attr_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(attr_frame, textvariable=self.face_attr_direction, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(attr_frame, text="Browse", command=self.browse_face_attr).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Face Attribute Steps
        steps_frame = ttk.Frame(adv_frame)
        steps_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(steps_frame, text="Face Attribute Steps:").pack(side=tk.LEFT)
        steps_spinbox = ttk.Spinbox(steps_frame, from_=-5.0, to=5.0, increment=0.1, 
                                   textvariable=self.face_attr_steps, width=10, format="%.1f")
        steps_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Options Frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="Enable OBS Virtual Camera", 
                       variable=self.obs_enabled).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Retain Target Mouth", 
                       variable=self.mouth_mask).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Show FPS and Delay", 
                       variable=self.fps_delay).pack(anchor=tk.W, pady=2)
        
        # Control Buttons Frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=20)
        
        self.start_button = ttk.Button(control_frame, text="Start Face Swap", 
                                      command=self.start_face_swap, style="Success.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Face Swap", 
                                     command=self.stop_face_swap, state=tk.DISABLED, style="Danger.TButton")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame, text="Test Webcam", command=self.test_webcam).pack(side=tk.LEFT)
        
        # Status Frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready to start", foreground="green")
        self.status_label.pack(anchor=tk.W)
        
        # Help Frame
        help_frame = ttk.LabelFrame(main_frame, text="Controls (while running)", padding=10)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        help_text = """• Press 'q' to quit
• Press '+' or '=' to increase delay by 50ms
• Press '-' to decrease delay by 50ms"""
        ttk.Label(help_frame, text=help_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Configure button styles
        style = ttk.Style()
        style.configure("Success.TButton", foreground="green")
        style.configure("Danger.TButton", foreground="red")
    
    def browse_source(self):
        filename = filedialog.askopenfilename(
            title="Select Source Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.source_path.set(filename)
    
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("PyTorch files", "*.pth *.pt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.model_path.set(filename)
    
    def browse_face_attr(self):
        filename = filedialog.askopenfilename(
            title="Select Face Attribute Direction File",
            filetypes=[
                ("NumPy files", "*.npy"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.face_attr_direction.set(filename)
    
    def validate_inputs(self):
        if not self.source_path.get():
            messagebox.showerror("Error", "Please select a source face image")
            return False
        
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file")
            return False
        
        if not os.path.exists(self.source_path.get()):
            messagebox.showerror("Error", "Source image file does not exist")
            return False
        
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Model file does not exist")
            return False
        
        if self.face_attr_direction.get() and not os.path.exists(self.face_attr_direction.get()):
            messagebox.showerror("Error", "Face attribute direction file does not exist")
            return False
        
        return True
    
    def build_command(self):
        # Assuming your original script is named 'face_swap.py'
        script_name = "swap_live_video.py"  # Change this to your actual script name
        
        cmd = [sys.executable, script_name]
        cmd.extend(["--source", self.source_path.get()])
        cmd.extend(["--modelPath", self.model_path.get()])
        cmd.extend(["--resolution", str(self.resolution.get())])
        cmd.extend(["--delay", str(self.delay.get())])
        
        if self.face_attr_direction.get():
            cmd.extend(["--face_attribute_direction", self.face_attr_direction.get()])
            cmd.extend(["--face_attribute_steps", str(self.face_attr_steps.get())])
        
        if self.obs_enabled.get():
            cmd.append("--obs")
        
        if self.mouth_mask.get():
            cmd.append("--mouth_mask")
        
        if self.fps_delay.get():
            cmd.append("--fps_delay")
        
        return cmd
    
    def start_face_swap(self):
        if not self.validate_inputs():
            return
        
        try:
            cmd = self.build_command()
            self.process = subprocess.Popen(cmd, 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE,
                                          text=True)
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Face swap is running...", foreground="blue")
            
            # Start monitoring thread
            threading.Thread(target=self.monitor_process, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start face swap: {str(e)}")
            self.reset_ui()
    
    def stop_face_swap(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"Error stopping process: {e}")
        
        self.reset_ui()
    
    def reset_ui(self):
        self.is_running = False
        self.process = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready to start", foreground="green")
    
    def monitor_process(self):
        if self.process:
            return_code = self.process.wait()
            if self.is_running:  # Only update if we didn't manually stop
                if return_code == 0:
                    self.status_label.config(text="Face swap finished", foreground="green")
                else:
                    self.status_label.config(text="Face swap stopped with error", foreground="red")
                self.reset_ui()
    
    def test_webcam(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Webcam Test", "Could not open webcam")
                return
            
            ret, frame = cap.read()
            if ret:
                messagebox.showinfo("Webcam Test", "Webcam is working properly!")
            else:
                messagebox.showerror("Webcam Test", "Could not read from webcam")
            
            cap.release()
        except Exception as e:
            messagebox.showerror("Webcam Test", f"Error testing webcam: {str(e)}")
    
    def on_closing(self):
        if self.is_running:
            if messagebox.askokcancel("Quit", "Face swap is running. Do you want to stop it and quit?"):
                self.stop_face_swap()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceSwapGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()