import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandVisualizer:
    def __init__(self, model_path=".venv\Lib\site-packages\hand_landmarker.task", max_hands=2):
        """Inisialisasi hand landmark detector"""
        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=0.3
            )
        )
        
        # Konfigurasi warna untuk garis
        self.COLOR_SCHEMES = {
            'vibrant': [
                (0, 255, 255),   # Kuning
                (255, 0, 0),     # Biru
                (0, 255, 0),     # Hijau
                (255, 0, 255),   # Magenta
                (0, 165, 255)    # Jingga
            ],
            'pastel': [
                (180, 255, 255), # Kuning pastel
                (255, 180, 180), # Merah muda
                (180, 255, 180), # Hijau muda
                (255, 180, 255), # Ungu muda
                (180, 200, 255)  # Biru muda
            ],
            'mono': [
                (200, 200, 200), # Abu-abu terang
                (150, 150, 150), # Abu-abu sedang
                (100, 100, 100), # Abu-abu gelap
                (200, 200, 200),
                (150, 150, 150)
            ]
        }
        
        # Koneksi antar titik (membentuk skeleton tangan)
        self.CONNECTIONS = [
            # Garis jari-jari (setiap jari 3 segmen)
            (1, 2), (2, 3), (3, 4),        # Ibu jari
            (5, 6), (6, 7), (7, 8),        # Telunjuk
            (9, 10), (10, 11), (11, 12),   # Jari tengah
            (13, 14), (14, 15), (15, 16),  # Jari manis
            (17, 18), (18, 19), (19, 20),  # Kelingking
            
            # Garis penghubung di telapak
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            
            # Garis melengkung di telapak
            (1, 5), (5, 9), (9, 13), (13, 17)
        ]
        
        self.color_scheme = 'vibrant'
        self.line_thickness = 2
    
    def set_color_scheme(self, scheme_name):
        """Ganti skema warna"""
        if scheme_name in self.COLOR_SCHEMES:
            self.color_scheme = scheme_name
    
    def set_line_thickness(self, thickness):
        """Atur ketebalan garis"""
        self.line_thickness = max(1, min(thickness, 5))
    
    def _get_finger_color(self, start_idx):
        """Dapatkan warna berdasarkan jari"""
        colors = self.COLOR_SCHEMES[self.color_scheme]
        
        if 1 <= start_idx <= 4:    # Ibu jari
            return colors[0]
        elif 5 <= start_idx <= 8:  # Telunjuk
            return colors[1]
        elif 9 <= start_idx <= 12: # Jari tengah
            return colors[2]
        elif 13 <= start_idx <= 16: # Jari manis
            return colors[3]
        elif 17 <= start_idx <= 20: # Kelingking
            return colors[4]
        else:                       # Telapak
            return (255, 255, 255)  # Putih
    
    def process_frame(self, frame):
        """Proses satu frame dan gambar garis-garis tangan"""
        # Konversi ke RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Deteksi tangan
        result = self.detector.detect(mp_image)
        
        if not result.hand_landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Gambar untuk setiap tangan
        for hand_landmarks in result.hand_landmarks:
            # Gambar semua garis koneksi
            for start_idx, end_idx in self.CONNECTIONS:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    # Konversi koordinat
                    x1 = int(hand_landmarks[start_idx].x * w)
                    y1 = int(hand_landmarks[start_idx].y * h)
                    x2 = int(hand_landmarks[end_idx].x * w)
                    y2 = int(hand_landmarks[end_idx].y * h)
                    
                    # Dapatkan warna
                    color = self._get_finger_color(start_idx)
                    
                    # Gambar garis
                    cv2.line(frame, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Opsional: gambar titik-titik kecil di persimpangan
            for idx, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Titik kecil di setiap landmark
                point_size = 7
                point_color = self._get_finger_color(idx)
                cv2.circle(frame, (x, y), point_size, point_color, -1)
        
        return frame

# ========== MAIN EXECUTION ==========
def main():
    # Buat visualizer
    visualizer = HandVisualizer()
    
    # Buka kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✋ Hand Visualizer - Garis Warna Saja")
    print("Kontrol:")
    print("  Q = Keluar")
    print("  1 = Skema Warna Vibrant")
    print("  2 = Skema Warna Pastel")
    print("  3 = Skema Warna Monokrom")
    print("  + = Pertebal garis")
    print("  - = Perkecil garis")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip untuk mirror effect
        frame = cv2.flip(frame, 1)
        
        # Proses frame
        output = visualizer.process_frame(frame)
        
        # Tampilkan
        cv2.imshow('Hand Visualizer - Garis Warna', output)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            visualizer.set_color_scheme('vibrant')
            print("Skema: Vibrant")
        elif key == ord('2'):
            visualizer.set_color_scheme('pastel')
            print("Skema: Pastel")
        elif key == ord('3'):
            visualizer.set_color_scheme('mono')
            print("Skema: Monokrom")
        elif key == ord('+'):
            visualizer.set_line_thickness(visualizer.line_thickness + 1)
            print(f"Ketebalan: {visualizer.line_thickness}")
        elif key == ord('-'):
            visualizer.set_line_thickness(visualizer.line_thickness - 1)
            print(f"Ketebalan: {visualizer.line_thickness}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()