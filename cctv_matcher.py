import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN

class CCTVMatcher:
    """
    Handles extracting frames from a CCTV video, detecting faces, 
    and matching them against a target suspect embedding.
    """
    def __init__(self, encoder, device=None):
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.encoder = encoder
        
        # MTCNN for face detection and cropping
        # keep_all=True allows detecting multiple faces in one frame
        # Adaptive Pooling on MPS often fails for MTCNN, so force face detection to CPU
        self.mtcnn = MTCNN(
            keep_all=True, 
            device="cpu",
            min_face_size=20,          # Catch smaller/background faces (default is 20)
            thresholds=[0.4, 0.5, 0.5] # Lowered thresholds from [0.6, 0.7, 0.7] to catch faces in bad lighting/CCTV quality
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def scan_video(self, video_path: str, target_embedding: torch.Tensor, threshold: float = 0.55, frame_skip: int = 15):
        """
        Scans a video for faces and compares them to the target embedding.
        
        Args:
            video_path: Path to the video file
            target_embedding: The FaceNet embedding of the suspect (1, 512)
            threshold: Cosine similarity threshold for a match
            frame_skip: How many frames to skip between checks (reduces processing time)
            
        Returns:
            list of dicts containing timestamp, similarity score, and cropped face image
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0 # fallback

        target_norm = torch.nn.functional.normalize(target_embedding, p=2, dim=1)
        matches = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process one frame every `frame_skip` frames
            if frame_count % frame_skip == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Detect faces in frame
                # MTCNN returns a list of cropped faces (as tensors) and probabilities
                # We use return_prob=False and instead just get the bounding boxes 
                # so we can crop them ourselves and run them through our existing encoder pipeline
                boxes, _ = self.mtcnn.detect(pil_img)
                
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box 
                        x1, y1, x2, y2 = [int(b) for b in box]
                        
                        # Add a small margin
                        margin = 20
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(pil_img.width, x2 + margin)
                        y2 = min(pil_img.height, y2 + margin)
                        
                        face_crop = pil_img.crop((x1, y1, x2, y2))
                        
                        # Skip tiny false positives
                        if face_crop.width < 40 or face_crop.height < 40:
                            continue
                            
                        # Encode face
                        face_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            face_emb = self.encoder.get_embedding(face_tensor)
                            
                        face_norm = torch.nn.functional.normalize(face_emb, p=2, dim=1)
                        
                        # Compare to suspect
                        sim = torch.mm(target_norm, face_norm.t()).item()
                        
                        if sim >= threshold:
                            timestamp_sec = frame_count / fps
                            minutes = int(timestamp_sec // 60)
                            seconds = int(timestamp_sec % 60)
                            
                            matches.append({
                                "timestamp": f"{minutes:02d}:{seconds:02d}",
                                "frame_number": frame_count,
                                "score": float(sim),
                                "pil_crop": face_crop, # Store PIL to convert to b64 later
                            })
                            
            frame_count += 1
            
        cap.release()
        
        # Sort matches by highest similarity score
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches
