import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import pickle
from PIL import Image, ImageTk

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x700")
        
        # Initialize variables
        self.known_face_embeddings = []
        self.known_face_names = []
        self.attendance_list = []
        self.class_names = []
        self.load_data()
        
        # Create GUI elements
        self.create_widgets()
        
        # Video capture
        self.video_capture = cv2.VideoCapture(0)
        self.video_running = False
        
    def create_widgets(self):
        # Main frames
        self.left_frame = tk.Frame(self.root, bg="white", width=400)
        self.left_frame.pack(side="left", fill="both", expand=True)
        
        self.right_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.right_frame.pack(side="right", fill="both", expand=True)
        
        # Left frame - Camera and controls
        self.camera_label = tk.Label(self.left_frame, bg="black")
        self.camera_label.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.control_frame = tk.Frame(self.left_frame, bg="white")
        self.control_frame.pack(pady=10, fill="x")
        
        self.start_btn = tk.Button(self.control_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = tk.Button(self.control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side="left", padx=5)
        
        self.register_btn = tk.Button(self.control_frame, text="Register Face", command=self.register_face)
        self.register_btn.pack(side="left", padx=5)
        
        self.mark_attendance_btn = tk.Button(self.control_frame, text="Mark Attendance", command=self.mark_attendance)
        self.mark_attendance_btn.pack(side="left", padx=5)
        
        # Right frame - Student list and attendance
        self.student_list_frame = tk.LabelFrame(self.right_frame, text="Class Students", padx=10, pady=10)
        self.student_list_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.tree = ttk.Treeview(self.student_list_frame, columns=("Name", "Status"), show="headings")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Status", text="Attendance Status")
        self.tree.column("Name", width=200)
        self.tree.column("Status", width=150)
        self.tree.pack(fill="both", expand=True)
        
        self.refresh_student_list()
        
        self.attendance_controls = tk.Frame(self.right_frame)
        self.attendance_controls.pack(pady=10, fill="x")
        
        self.export_btn = tk.Button(self.attendance_controls, text="Export to Excel", command=self.export_to_excel)
        self.export_btn.pack(side="left", padx=5)
        
        self.clear_btn = tk.Button(self.attendance_controls, text="Clear Attendance", command=self.clear_attendance)
        self.clear_btn.pack(side="left", padx=5)
        
        self.attendance_status = tk.Label(self.right_frame, text="Attendance not marked yet", font=("Arial", 10))
        self.attendance_status.pack(pady=5)
        
    def load_data(self):
        # Load known faces if data file exists
        if os.path.exists("face_data.pkl"):
            with open("face_data.pkl", "rb") as f:
                data = pickle.load(f)
                self.known_face_embeddings = data["embeddings"]
                self.known_face_names = data["names"]
                self.class_names = list(set(self.known_face_names))
        
    def save_data(self):
        # Save known faces to file
        data = {
            "embeddings": self.known_face_embeddings,
            "names": self.known_face_names
        }
        with open("face_data.pkl", "wb") as f:
            pickle.dump(data, f)
    
    def refresh_student_list(self):
        # Refresh the student list in the treeview
        self.tree.delete(*self.tree.get_children())
        self.class_names = list(set(self.known_face_names))
        
        for name in sorted(self.class_names):
            status = "Present" if name in self.attendance_list else "Absent"
            self.tree.insert("", "end", values=(name, status))
    
    def start_camera(self):
        if not self.video_running:
            self.video_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.show_camera()
    
    def stop_camera(self):
        self.video_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def show_camera(self):
        if self.video_running:
            ret, frame = self.video_capture.read()
            if ret:
                # Convert to RGB for DeepFace
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Find all face locations and embeddings
                    face_objs = DeepFace.extract_faces(rgb_frame, detector_backend='opencv', enforce_detection=False)
                    
                    for face_obj in face_objs:
                        if face_obj['confidence'] > 0.9:  # Only consider faces with high confidence
                            facial_area = face_obj['facial_area']
                            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                            
                            # Get face embedding
                            face_img = rgb_frame[y:y+h, x:x+w]
                            try:
                                embedding_obj = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
                                if embedding_obj and len(embedding_obj) > 0:
                                    face_embedding = embedding_obj[0]['embedding']
                                    
                                    # Compare with known faces
                                    if self.known_face_embeddings:
                                        result = DeepFace.verify(
                                            img1_path=face_embedding,
                                            img2_path=self.known_face_embeddings,
                                            model_name='Facenet',
                                            distance_metric='cosine',
                                            enforce_detection=False
                                        )
                                        
                                        # Find the best match
                                        best_match_index = np.argmin(result['distance'])
                                        if result['distance'][best_match_index] < 0.4:  # Threshold for match
                                            name = self.known_face_names[best_match_index]
                                        else:
                                            name = "Unknown"
                                    else:
                                        name = "Unknown"
                                    
                                    # Draw rectangle and label
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(frame, name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            except Exception as e:
                                print(f"Error processing face: {e}")
                except Exception as e:
                    print(f"Face detection error: {e}")
                
                # Convert to PhotoImage and display
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
                
                # Repeat after 10ms
                self.camera_label.after(10, self.show_camera)
    
    def register_face(self):
        if not self.video_running:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
        
        # Create registration window
        register_window = tk.Toplevel(self.root)
        register_window.title("Register New Student")
        register_window.geometry("400x200")
        
        tk.Label(register_window, text="Student Name:").pack(pady=10)
        name_entry = tk.Entry(register_window, width=30)
        name_entry.pack(pady=5)
        
        def capture_face():
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a name")
                return
            
            ret, frame = self.video_capture.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Detect face using DeepFace
                    face_objs = DeepFace.extract_faces(rgb_frame, detector_backend='opencv', enforce_detection=True)
                    
                    if len(face_objs) == 0:
                        messagebox.showerror("Error", "No face detected. Please try again.")
                        return
                    
                    # Get the first face
                    face_obj = face_objs[0]
                    if face_obj['confidence'] > 0.9:
                        facial_area = face_obj['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                        face_img = rgb_frame[y:y+h, x:x+w]
                        
                        # Get face embedding
                        embedding_obj = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
                        if embedding_obj and len(embedding_obj) > 0:
                            face_embedding = embedding_obj[0]['embedding']
                            
                            # Add to known faces
                            self.known_face_embeddings.append(face_embedding)
                            self.known_face_names.append(name)
                            self.save_data()
                            
                            messagebox.showinfo("Success", f"Face registered for {name}")
                            register_window.destroy()
                            self.refresh_student_list()
                        else:
                            messagebox.showerror("Error", "Could not extract face features. Please try again.")
                    else:
                        messagebox.showerror("Error", "Low confidence face detection. Please try again.")
                except Exception as e:
                    messagebox.showerror("Error", f"Face detection failed: {str(e)}")
        
        capture_btn = tk.Button(register_window, text="Capture Face", command=capture_face)
        capture_btn.pack(pady=20)
    
    def mark_attendance(self):
        if not self.video_running:
            messagebox.showwarning("Warning", "Please start the camera first")
            return
        
        self.attendance_list = []
        ret, frame = self.video_capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Detect all faces
                face_objs = DeepFace.extract_faces(rgb_frame, detector_backend='opencv', enforce_detection=False)
                
                for face_obj in face_objs:
                    if face_obj['confidence'] > 0.9:  # Only consider faces with high confidence
                        facial_area = face_obj['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                        face_img = rgb_frame[y:y+h, x:x+w]
                        
                        # Get face embedding
                        embedding_obj = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
                        if embedding_obj and len(embedding_obj) > 0:
                            face_embedding = embedding_obj[0]['embedding']
                            
                            # Compare with known faces
                            if self.known_face_embeddings:
                                result = DeepFace.verify(
                                    img1_path=face_embedding,
                                    img2_path=self.known_face_embeddings,
                                    model_name='Facenet',
                                    distance_metric='cosine',
                                    enforce_detection=False
                                )
                                
                                # Find the best match
                                best_match_index = np.argmin(result['distance'])
                                if result['distance'][best_match_index] < 0.4:  # Threshold for match
                                    name = self.known_face_names[best_match_index]
                                    if name not in self.attendance_list:
                                        self.attendance_list.append(name)
            except Exception as e:
                print(f"Error during attendance marking: {e}")
            
            self.refresh_student_list()
            self.attendance_status.config(text=f"Attendance marked for {len(self.attendance_list)} students")
    
    def export_to_excel(self):
        if not self.attendance_list:
            messagebox.showwarning("Warning", "No attendance data to export")
            return
        
        # Create a DataFrame with all students and their status
        all_students = sorted(list(set(self.known_face_names)))
        status = ["Present" if student in self.attendance_list else "Absent" for student in all_students]
        date = datetime.now().strftime("%Y-%m-%d")
        
        df = pd.DataFrame({
            "Name": all_students,
            "Status": status,
            "Date": date
        })
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Save attendance file"
        )
        
        if file_path:
            try:
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", f"Attendance exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def clear_attendance(self):
        self.attendance_list = []
        self.refresh_student_list()
        self.attendance_status.config(text="Attendance cleared")
    
    def on_closing(self):
        self.video_running = False
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()