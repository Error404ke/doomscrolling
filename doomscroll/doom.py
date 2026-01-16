import cv2
import numpy as np
import time
import threading
import webbrowser
import pyautogui
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from datetime import datetime
import json
import winsound
import platform
import sys
import subprocess
import random


def install_packages():
    required_packages = {
        'opencv-python': 'cv2',
        'pyautogui': 'pyautogui',
        'Pillow': 'PIL'
    }
    
    print("Checking dependencies...")
    for package, import_name in required_packages.items():
        try:
            if import_name == 'PIL':
                __import__('PIL')
            else:
                __import__(import_name)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except:
                print(f"✗ Failed to install {package}")

# Install packages first
install_packages()


try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
    print("✓ PyGame available for audio playback")
except:
    PYGAME_AVAILABLE = False
    print("✗ PyGame not available, using fallback sounds")

try:
    from plyer import notification
    NOTIFICATION_AVAILABLE = True
except:
    NOTIFICATION_AVAILABLE = False
    print("✗ Plyer not available, notifications limited")

class SimpleScrollDetector:
    """Simplified scroll detector using basic computer vision"""
    
    def __init__(self, sound_file=None):
        # Detection parameters
        self.scroll_threshold = 20
        self.swipe_threshold = 50
        self.focus_time_threshold = 15
        self.doomscroll_time_threshold = 5
        

        self.sound_file = sound_file
        self.sound_directory = os.path.dirname(os.path.abspath(__file__))
        self.default_sounds = []
        
       
        self.load_default_sounds()
        
  
        self.last_hand_positions = {'left': None, 'right': None}
        self.last_scroll_time = 0
        self.scroll_count = 0
        self.doomscroll_start_time = None
        self.focus_start_time = time.time()
        self.last_warning_time = 0
        self.last_movement_time = time.time()
        
      
        self.doomscrolling = False
        self.rickrolled = False
        self.running = True
        self.paused = False
        
       
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cascade_available = True
        except:
            self.face_cascade_available = False
            print("Warning: Face cascade not available, using motion detection only")
        
  
        self.prev_gray = None
        self.motion_threshold = 1000
        
        
        self.stats = {
            'total_focus_time': 0,
            'doomscroll_sessions': 0,
            'warnings_given': 0,
            'rickrolls_triggered': 0,
            'last_reset': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
       
        self.load_stats()
    
    def load_default_sounds(self):
        """Load default sound files from directory"""
        sound_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
        
       
        for file in os.listdir(self.sound_directory):
            if any(file.lower().endswith(ext) for ext in sound_extensions):
                self.default_sounds.append(os.path.join(self.sound_directory, file))
        
       
        sounds_dir = os.path.join(self.sound_directory, 'sounds')
        if os.path.exists(sounds_dir):
            for file in os.listdir(sounds_dir):
                if any(file.lower().endswith(ext) for ext in sound_extensions):
                    self.default_sounds.append(os.path.join(sounds_dir, file))
        
        print(f"Found {len(self.default_sounds)} sound files in directory")
        for sound in self.default_sounds:
            print(f"  - {os.path.basename(sound)}")
    
    def load_stats(self):
        """Load statistics from file"""
        try:
            if os.path.exists('doomscroll_stats.json'):
                with open('doomscroll_stats.json', 'r') as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
        except Exception as e:
            print(f"Error loading stats: {e}")
    
    def save_stats(self):
        """Save statistics to file"""
        try:
            with open('doomscroll_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            print(f"Error saving stats: {e}")
    
    def set_sound_file(self, sound_file):
        """Set the sound file to play"""
        if sound_file and os.path.exists(sound_file):
            self.sound_file = sound_file
            return True
        return False
    
    def play_random_sound(self):
        """Play a random sound from available sounds"""
        if self.default_sounds:
            sound_file = random.choice(self.default_sounds)
            self.play_sound_file(sound_file)
        else:
            self.play_fallback_sound()
    
    def play_sound_file(self, sound_file):
        """Play a specific sound file"""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()
                print(f"Playing sound: {os.path.basename(sound_file)}")
                
              
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error playing sound {sound_file}: {e}")
                self.play_fallback_sound()
        else:
            self.play_fallback_sound()
    
    def detect_motion(self, frame):
        """Detect motion in the frame using optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0
        
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
     
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_score = np.mean(magnitude) * 1000
        
        self.prev_gray = gray
        
        return motion_score > self.motion_threshold, motion_score
    
    def detect_hand_movement(self, frame):
        """Simple hand movement detection using contour detection"""
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
      
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
     
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
     
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        
    
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
      
        hand_detected = False
        max_area = 0
        hand_center = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000 and area > max_area:  # Minimum area for hand
                max_area = area
                hand_detected = True
                
             
                x, y, w, h = cv2.boundingRect(contour)
                hand_center = (x + w // 2, y + h // 2)
                
             
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return hand_detected, hand_center
    
    def detect_face(self, frame):
        """Detect face in frame"""
        if not self.face_cascade_available:
            return False, []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_detected = len(faces) > 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        return face_detected, faces
    
    def trigger_rickroll(self):
        """Trigger the RickRoll experience"""
        self.rickrolled = True
        self.stats['rickrolls_triggered'] += 1
        self.stats['doomscroll_sessions'] += 1
        self.save_stats()
        
        
        threading.Thread(target=self.play_alert_sound, daemon=True).start()
        
       
        threading.Thread(target=self.open_rickroll, daemon=True).start()
        
       
        threading.Thread(target=self.show_popup, daemon=True).start()
        
      
        self.show_notification("DOOMSCROLL DETECTED!", "You've been RickRolled! Get back to work!")
        
     
        if platform.system() == "Windows":
            for _ in range(3):
                winsound.Beep(1000, 500)
                time.sleep(0.2)
    
    def play_alert_sound(self):
        """Play alert sound from directory"""
        if self.sound_file and os.path.exists(self.sound_file):
            print(f"Playing custom sound: {self.sound_file}")
            self.play_sound_file(self.sound_file)
        elif self.default_sounds:
            print("Playing random sound from directory")
            self.play_random_sound()
        else:
            print("No sound files found, playing fallback sound")
            self.play_fallback_sound()
    
    def play_fallback_sound(self):
        """Fallback sound using system beeps"""
        if platform.system() == "Windows":
            print("Playing Windows beep sound")
            for freq in [523, 587, 659, 698, 784, 880, 988, 1047]:
                winsound.Beep(freq, 200)
                time.sleep(0.15)
        else:
            print("Playing simple beep pattern")
          
            for _ in range(5):
                print("\a", end='', flush=True)
                time.sleep(0.2)
    
    def open_rickroll(self):
        """Open RickRoll video in browser"""
        try:
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ", new=2)
            time.sleep(1)
            
            # Try to make it fullscreen
            pyautogui.press('f')
        except:
            pass
    
    def show_popup(self):
        """Show popup warning"""
        try:
            popup = tk.Tk()
            popup.title("🚨 DOOMSCROLL DETECTED! 🚨")
            popup.geometry("500x300")
            popup.configure(bg='red')
            popup.attributes('-topmost', True)
            
            label = tk.Label(
                popup,
                text="🎵 NEVER GONNA GIVE YOU UP! 🎵\n\n"
                     "You've been RickRolled for doomscrolling!\n\n"
                     "Get back to work!\n\n"
                     "This window will close in 20 seconds...",
                font=('Arial', 16, 'bold'),
                fg='yellow',
                bg='red',
                pady=30
            )
            label.pack(expand=True)
            
            countdown_label = tk.Label(
                popup,
                text="20",
                font=('Arial', 36, 'bold'),
                fg='white',
                bg='red'
            )
            countdown_label.pack()
            
            def update_countdown(seconds=20):
                if seconds > 0:
                    countdown_label.config(text=str(seconds))
                    popup.after(1000, update_countdown, seconds-1)
                else:
                    popup.destroy()
                    self.rickrolled = False
            
            update_countdown()
            popup.mainloop()
        except:
            self.rickrolled = False
    
    def show_notification(self, title, message):
        """Show system notification"""
        if NOTIFICATION_AVAILABLE:
            try:
                notification.notify(
                    title=title,
                    message=message,
                    app_name="DoomScroll Detector",
                    timeout=5
                )
            except:
                pass
    
    def show_warning(self):
        """Show warning before RickRoll"""
        current_time = time.time()
        if current_time - self.last_warning_time > 10:
            self.last_warning_time = current_time
            self.stats['warnings_given'] += 1
            
            try:
                # Simple warning in console
                print("⚠️  WARNING: You're scrolling too much! Get back to work or you'll be RickRolled!")
                
                # Flash taskbar (Windows)
                if platform.system() == "Windows":
                    flash_window = tk.Tk()
                    flash_window.withdraw()
                    flash_window.attributes('-topmost', True)
                    flash_window.update()
                    flash_window.attributes('-topmost', False)
                    flash_window.destroy()
            except:
                pass
    
    def process_frame(self, frame):
        """Process a single frame for doomscroll detection"""
        if self.paused or self.rickrolled:
            return frame
        
        output_frame = frame.copy()
        
   
        face_detected, faces = self.detect_face(output_frame)
        
   
        hand_detected, hand_center = self.detect_hand_movement(output_frame)
        
     
        motion_detected, motion_score = self.detect_motion(frame)
        
      
        current_time = time.time()
        
      
        if motion_detected or hand_detected:
            self.last_movement_time = current_time
        
    
        movement_duration = current_time - self.last_movement_time
        
        # If lots of movement but no face (looking at phone)
        if (motion_detected or hand_detected) and not face_detected:
            if self.doomscroll_start_time is None:
                self.doomscroll_start_time = current_time
                self.doomscrolling = True
            
            doomscroll_duration = current_time - self.doomscroll_start_time
            
            # Show warning if approaching threshold
            if doomscroll_duration > self.doomscroll_time_threshold - 2:
                self.show_warning()
            
            # Trigger RickRoll if threshold exceeded
            if doomscroll_duration > self.doomscroll_time_threshold and not self.rickrolled:
                self.trigger_rickroll()
                self.doomscroll_start_time = None
        else:
            # Reset doomscroll timer
            self.doomscroll_start_time = None
            self.doomscrolling = False
            
            # Track focus time
            if face_detected:
                self.stats['total_focus_time'] += 1
                if current_time - self.focus_start_time > self.focus_time_threshold:
                    # Reward for good behavior
                    if not self.rickrolled:
                        self.show_notification("Great Focus!", 
                                             f"You've been focused for {self.focus_time_threshold} seconds!")
                        self.focus_start_time = current_time
        
        # Add overlay
        self.add_status_overlay(output_frame, face_detected, hand_detected, 
                               motion_detected, self.doomscrolling)
        
        return output_frame
    
    def add_status_overlay(self, frame, face_detected, hand_detected, motion_detected, doomscrolling):
        """Add status overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status texts
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        status_text = "FOCUSED ✓" if face_detected else "DISTRACTED ✗"
        
        cv2.putText(frame, f"Status: {status_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(frame, f"Hand Detected: {'YES' if hand_detected else 'NO'}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if hand_detected else (255, 255, 255), 2)
        
        cv2.putText(frame, f"Motion: {'HIGH' if motion_detected else 'LOW'}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if motion_detected else (255, 255, 255), 2)
        
        if doomscrolling:
            doom_time = int(time.time() - self.doomscroll_start_time) if self.doomscroll_start_time else 0
            warning_text = f"DOOMSCROLLING! {doom_time}s"
            cv2.putText(frame, warning_text, (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add watermark
        cv2.putText(frame, "DoomScroll Detector v1.0", (w - 250, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show current sound file
        if self.sound_file:
            sound_name = os.path.basename(self.sound_file)
            cv2.putText(frame, f"Sound: {sound_name[:15]}...", (w - 250, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
    
    def run(self):
        """Main loop for webcam processing"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            messagebox.showerror("Error", "Could not open webcam. Please check your camera connection.")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*50)
        print("DOOMSCROLL DETECTOR STARTED!")
        print("="*50)
        
        if self.sound_file:
            print(f"Sound file: {os.path.basename(self.sound_file)}")
        elif self.default_sounds:
            print(f"Using {len(self.default_sounds)} sound files from directory")
        else:
            print("No sound files found, using system beeps")
            
        print("\nControls:")
        print("  - Press 'p' to pause/resume")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset counters")
        print("\nStay focused and avoid excessive movement!")
        print("="*50 + "\n")
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Show frame
            cv2.imshow('DoomScroll Detector - Stay Focused!', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nShutting down...")
                break
            elif key == ord('p'):
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RUNNING"
                print(f"\nStatus: {status}")
            elif key == ord('r'):
                self.scroll_count = 0
                self.doomscroll_start_time = None
                self.focus_start_time = time.time()
                print("\nCounters reset!")
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_stats()


class DoomScrollApp:
    """GUI Application for DoomScroll Detector"""
    
    def __init__(self, root):
        self.root = root
        self.sound_file = None
        self.detector = SimpleScrollDetector()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.root.title("DoomScroll Detector & RickRoller")
        self.root.geometry("650x750")
        self.root.configure(bg='#2c3e50')
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Header
        header = tk.Label(
            self.root,
            text="🚀 DoomScroll Detector",
            font=('Arial', 24, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50',
            pady=20
        )
        header.pack()
        
        # Subtitle
        subtitle = tk.Label(
            self.root,
            text="Get RickRolled if you doomscroll while working!",
            font=('Arial', 12, 'italic'),
            fg='#f1c40f',
            bg='#2c3e50'
        )
        subtitle.pack()
        
        # Sound Selection Frame
        sound_frame = tk.Frame(self.root, bg='#34495e', padx=20, pady=15, relief='ridge', bd=2)
        sound_frame.pack(pady=10, fill='x', padx=30)
        
        tk.Label(
            sound_frame,
            text="🔊 Sound Settings",
            font=('Arial', 16, 'bold'),
            fg='#1abc9c',
            bg='#34495e'
        ).pack(anchor='w', pady=(0, 10))
        
        # Sound file display
        self.sound_label = tk.Label(
            sound_frame,
            text="No sound file selected",
            font=('Arial', 10),
            fg='#bdc3c7',
            bg='#34495e',
            wraplength=500,
            justify='left'
        )
        self.sound_label.pack(fill='x', pady=5)
        
        # Sound buttons frame
        sound_buttons_frame = tk.Frame(sound_frame, bg='#34495e')
        sound_buttons_frame.pack(fill='x', pady=10)
        
        # Select Sound Button
        select_sound_btn = tk.Button(
            sound_buttons_frame,
            text="🎵 Select Sound File",
            command=self.select_sound_file,
            font=('Arial', 11),
            bg='#3498db',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        select_sound_btn.pack(side='left', padx=5)
        
        # Clear Sound Button
        clear_sound_btn = tk.Button(
            sound_buttons_frame,
            text="🗑️ Clear Sound",
            command=self.clear_sound_file,
            font=('Arial', 11),
            bg='#e74c3c',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        clear_sound_btn.pack(side='left', padx=5)
        
        # Test Sound Button
        test_sound_btn = tk.Button(
            sound_buttons_frame,
            text="▶️ Test Sound",
            command=self.test_selected_sound,
            font=('Arial', 11),
            bg='#2ecc71',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2'
        )
        test_sound_btn.pack(side='left', padx=5)
        
        # Available sounds label
        if self.detector.default_sounds:
            sounds_text = f"Found {len(self.detector.default_sounds)} sound files in directory:"
            tk.Label(
                sound_frame,
                text=sounds_text,
                font=('Arial', 9, 'bold'),
                fg='#ecf0f1',
                bg='#34495e',
                anchor='w'
            ).pack(fill='x', pady=(10, 0))
            
            # Show first few sounds
            for i, sound in enumerate(self.detector.default_sounds[:3]):
                sound_name = os.path.basename(sound)
                tk.Label(
                    sound_frame,
                    text=f"• {sound_name}",
                    font=('Arial', 8),
                    fg='#bdc3c7',
                    bg='#34495e',
                    anchor='w'
                ).pack(fill='x')
            
            if len(self.detector.default_sounds) > 3:
                tk.Label(
                    sound_frame,
                    text=f"... and {len(self.detector.default_sounds) - 3} more",
                    font=('Arial', 8, 'italic'),
                    fg='#7f8c8d',
                    bg='#34495e',
                    anchor='w'
                ).pack(fill='x')
        
        # Stats Frame
        stats_frame = tk.Frame(self.root, bg='#34495e', padx=20, pady=20, relief='ridge', bd=2)
        stats_frame.pack(pady=15, fill='x', padx=30)
        
        tk.Label(
            stats_frame,
            text="📊 Your Statistics",
            font=('Arial', 16, 'bold'),
            fg='#1abc9c',
            bg='#34495e'
        ).pack(anchor='w', pady=(0, 10))
        
        self.stats_labels = {}
        stats_data = [
            ("Total Focus Time", f"{self.detector.stats['total_focus_time']}s"),
            ("DoomScroll Sessions", str(self.detector.stats['doomscroll_sessions'])),
            ("Warnings Given", str(self.detector.stats['warnings_given'])),
            ("RickRolls Triggered", str(self.detector.stats['rickrolls_triggered']))
        ]
        
        for stat_name, stat_value in stats_data:
            frame = tk.Frame(stats_frame, bg='#34495e')
            frame.pack(fill='x', pady=3)
            
            tk.Label(
                frame,
                text=stat_name + ":",
                font=('Arial', 11),
                fg='#ecf0f1',
                bg='#34495e',
                width=25,
                anchor='w'
            ).pack(side='left')
            
            label = tk.Label(
                frame,
                text=stat_value,
                font=('Arial', 11, 'bold'),
                fg='#f1c40f',
                bg='#34495e',
                width=15,
                anchor='w'
            )
            label.pack(side='left')
            self.stats_labels[stat_name] = label
        
        # Controls Frame
        controls_frame = tk.Frame(self.root, bg='#2c3e50', pady=20)
        controls_frame.pack()
        
        # Start Button
        self.start_button = tk.Button(
            controls_frame,
            text="▶️ Start Detection",
            command=self.start_detection,
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=12,
            relief='raised',
            borderwidth=3,
            cursor='hand2'
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Pause Button
        self.pause_button = tk.Button(
            controls_frame,
            text="⏸️ Pause",
            command=self.toggle_pause,
            font=('Arial', 14),
            bg='#f39c12',
            fg='white',
            padx=20,
            pady=12,
            state='disabled',
            cursor='hand2'
        )
        self.pause_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Reset Button
        self.reset_button = tk.Button(
            controls_frame,
            text="🔄 Reset Stats",
            command=self.reset_stats,
            font=('Arial', 14),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=12,
            cursor='hand2'
        )
        self.reset_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Settings Frame
        settings_frame = tk.Frame(self.root, bg='#34495e', padx=20, pady=15, relief='ridge', bd=2)
        settings_frame.pack(pady=15, fill='x', padx=30)
        
        tk.Label(
            settings_frame,
            text="⚙️ Detection Settings",
            font=('Arial', 16, 'bold'),
            fg='#1abc9c',
            bg='#34495e'
        ).pack(anchor='w', pady=(0, 10))
        
        # Sensitivity setting
        sensitivity_frame = tk.Frame(settings_frame, bg='#34495e')
        sensitivity_frame.pack(fill='x', pady=5)
        
        tk.Label(
            sensitivity_frame,
            text="Sensitivity:",
            font=('Arial', 11),
            fg='#ecf0f1',
            bg='#34495e',
            width=15,
            anchor='w'
        ).pack(side='left')
        
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_slider = ttk.Scale(
            sensitivity_frame,
            from_=0.1,
            to=1.0,
            variable=self.sensitivity_var,
            orient='horizontal',
            command=self.update_sensitivity
        )
        sensitivity_slider.pack(side='left', fill='x', expand=True, padx=10)
        
        self.sensitivity_label = tk.Label(
            sensitivity_frame,
            text="Medium",
            font=('Arial', 11),
            fg='#f1c40f',
            bg='#34495e',
            width=10
        )
        self.sensitivity_label.pack(side='right')
        
        # Test RickRoll Button
        test_frame = tk.Frame(self.root, bg='#2c3e50', pady=15)
        test_frame.pack()
        
        test_button = tk.Button(
            test_frame,
            text="🎵 Test RickRoll (Preview)",
            command=self.test_rickroll,
            font=('Arial', 12),
            bg='#9b59b6',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        test_button.pack()
        
        # Instructions
        instructions_frame = tk.Frame(self.root, bg='#34495e', padx=20, pady=15, relief='ridge', bd=2)
        instructions_frame.pack(pady=15, fill='x', padx=30)
        
        tk.Label(
            instructions_frame,
            text="💡 How It Works:",
            font=('Arial', 14, 'bold'),
            fg='#1abc9c',
            bg='#34495e'
        ).pack(anchor='w', pady=(0, 5))
        
        instructions = [
            "1. Select a sound file to play when doomscrolling",
            "2. Sit facing your webcam",
            "3. Keep your face visible to the camera",
            "4. Avoid excessive movement (doomscrolling)",
            "5. If detected doomscrolling, your sound will play!",
            "6. Press 'q' in camera window to quit detection"
        ]
        
        for instruction in instructions:
            tk.Label(
                instructions_frame,
                text=instruction,
                font=('Arial', 10),
                fg='#ecf0f1',
                bg='#34495e',
                anchor='w'
            ).pack(fill='x', pady=2)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready to start detection",
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#34495e',
            relief='sunken',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side='bottom', fill='x')
        
        # Update stats periodically
        self.update_stats()
    
    def select_sound_file(self):
        """Select a sound file to play when doomscrolling"""
        filetypes = [
            ('Audio files', '*.mp3 *.wav *.ogg *.flac *.m4a *.aac'),
            ('MP3 files', '*.mp3'),
            ('WAV files', '*.wav'),
            ('All files', '*.*')
        ]
        
        sound_file = filedialog.askopenfilename(
            title="Select Sound File",
            filetypes=filetypes,
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        
        if sound_file:
            if self.detector.set_sound_file(sound_file):
                self.sound_file = sound_file
                sound_name = os.path.basename(sound_file)
                self.sound_label.config(
                    text=f"Selected: {sound_name}",
                    fg='#2ecc71'
                )
                self.status_bar.config(text=f"Sound file selected: {sound_name}", fg='#2ecc71')
                print(f"Sound file set to: {sound_file}")
            else:
                messagebox.showerror("Error", "Could not load sound file. Please select a valid audio file.")
    
    def clear_sound_file(self):
        """Clear the selected sound file"""
        self.sound_file = None
        self.detector.sound_file = None
        self.sound_label.config(text="No sound file selected", fg='#bdc3c7')
        self.status_bar.config(text="Sound file cleared", fg='#f39c12')
        print("Sound file cleared")
    
    def test_selected_sound(self):
        """Test the selected sound file"""
        if self.sound_file and os.path.exists(self.sound_file):
            self.status_bar.config(text="Testing sound file...", fg='#3498db')
            
            # Play sound in separate thread
            def play_test_sound():
                try:
                    if PYGAME_AVAILABLE:
                        pygame.mixer.music.load(self.sound_file)
                        pygame.mixer.music.play()
                        print(f"Testing sound: {os.path.basename(self.sound_file)}")
                        
                        # Wait for sound to finish
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        
                        self.root.after(0, lambda: self.status_bar.config(
                            text="Sound test complete!",
                            fg='#2ecc71'
                        ))
                    else:
                        messagebox.showwarning("Audio Error", 
                                             "PyGame not available. Cannot play sound files.")
                except Exception as e:
                    self.root.after(0, lambda: self.status_bar.config(
                        text=f"Error playing sound: {str(e)[:30]}...",
                        fg='#e74c3c'
                    ))
            
            threading.Thread(target=play_test_sound, daemon=True).start()
        elif self.detector.default_sounds:
            self.status_bar.config(text="Playing random sound from directory...", fg='#3498db')
            self.detector.play_random_sound()
            self.root.after(1000, lambda: self.status_bar.config(
                text="Sound test complete!",
                fg='#2ecc71'
            ))
        else:
            messagebox.showinfo("No Sound", "No sound file selected and no sounds found in directory.")
    
    def start_detection(self):
        """Start the doomscroll detection"""
        self.start_button.config(state='disabled', bg='#95a5a6', text="⏳ Starting...")
        self.pause_button.config(state='normal')
        self.reset_button.config(state='disabled')
        self.status_bar.config(text="Detection started - Monitoring your focus...", fg='#27ae60')
        
        # Pass sound file to detector
        if self.sound_file:
            self.detector.set_sound_file(self.sound_file)
        
        # Start detector in separate thread
        detector_thread = threading.Thread(target=self.detector.run)
        detector_thread.daemon = True
        detector_thread.start()
        
        # Update button after a short delay
        self.root.after(1000, lambda: self.start_button.config(
            text="✅ Detection Running",
            bg='#27ae60'
        ))
        
        # Re-enable reset button after 2 seconds
        self.root.after(2000, lambda: self.reset_button.config(state='normal'))
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.detector.paused = not self.detector.paused
        if self.detector.paused:
            self.pause_button.config(text="▶️ Resume", bg='#27ae60')
            self.status_bar.config(text="Detection PAUSED", fg='#f39c12')
        else:
            self.pause_button.config(text="⏸️ Pause", bg='#f39c12')
            self.status_bar.config(text="Detection RESUMED", fg='#27ae60')
    
    def reset_stats(self):
        """Reset all statistics"""
        response = messagebox.askyesno(
            "Reset Statistics",
            "Are you sure you want to reset all statistics?\nThis cannot be undone."
        )
        
        if response:
            self.detector.stats = {
                'total_focus_time': 0,
                'doomscroll_sessions': 0,
                'warnings_given': 0,
                'rickrolls_triggered': 0,
                'last_reset': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.detector.scroll_count = 0
            self.detector.doomscroll_start_time = None
            self.detector.focus_start_time = time.time()
            self.detector.save_stats()
            self.update_stats()
            self.status_bar.config(text="Statistics reset successfully!", fg='#f1c40f')
    
    def update_sensitivity(self, value):
        """Update detection sensitivity"""
        sensitivity = float(value)
        self.detector.scroll_threshold = int(30 * sensitivity)
        self.detector.swipe_threshold = int(60 * sensitivity)
        self.detector.doomscroll_time_threshold = 3 + (7 * (1 - sensitivity))
        self.detector.motion_threshold = 500 + (1500 * sensitivity)
        
        # Update label
        if sensitivity < 0.4:
            level = "Low"
        elif sensitivity < 0.7:
            level = "Medium"
        else:
            level = "High"
        self.sensitivity_label.config(text=level)
    
    def test_rickroll(self):
        """Test RickRoll functionality"""
        response = messagebox.askyesno(
            "Test RickRoll",
            "This will open RickRoll in your browser AND play your selected sound.\nDo you want to continue?"
        )
        
        if response:
            self.status_bar.config(text="Testing RickRoll...", fg='#9b59b6')
            
            # Play sound
            if self.sound_file and os.path.exists(self.sound_file):
                threading.Thread(target=self.detector.play_alert_sound, daemon=True).start()
            elif self.detector.default_sounds:
                threading.Thread(target=self.detector.play_random_sound, daemon=True).start()
            
            # Open browser
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ", new=2)
            
            self.root.after(3000, lambda: self.status_bar.config(
                text="RickRoll test complete!",
                fg='#27ae60'
            ))
    
    def update_stats(self):
        """Update statistics display"""
        if hasattr(self, 'detector') and hasattr(self.detector, 'stats'):
            stats = self.detector.stats
            
            # Update all stat labels
            self.stats_labels["Total Focus Time"].config(text=f"{stats['total_focus_time']}s")
            self.stats_labels["DoomScroll Sessions"].config(text=str(stats['doomscroll_sessions']))
            self.stats_labels["Warnings Given"].config(text=str(stats['warnings_given']))
            self.stats_labels["RickRolls Triggered"].config(text=str(stats['rickrolls_triggered']))
        
        # Schedule next update
        self.root.after(1000, self.update_stats)


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("        DOOMSCROLL DETECTOR & RICKROLLER")
    print("="*60)
    print("\n🎯 Purpose: Keep you focused while working/studying")
    print("🔊 Feature: Play custom sounds when doomscrolling")
    print("📊 Features: Focus tracking, statistics, warnings")
    print("\n" + "-"*60)
    
    # Create main window
    root = tk.Tk()
    app = DoomScrollApp(root)
    
    # Center window
    root.update_idletasks()
    width = 650
    height = 750
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            if hasattr(app, 'detector'):
                app.detector.running = False
                app.detector.save_stats()
            root.destroy()
            print("\nApplication closed. Goodbye!")
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Make window stay on top initially
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)
    
    print("\n✓ GUI loaded successfully!")
    print("✓ Click 'Select Sound File' to choose your custom sound")
    print("✓ Click 'Start Detection' to begin monitoring")
    print("="*60 + "\n")
    
    root.mainloop()


if __name__ == "__main__":
    main()