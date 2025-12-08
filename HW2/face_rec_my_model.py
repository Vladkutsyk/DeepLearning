import os
import cv2
import numpy as np
from face_recognition import face_locations  # keep dlib detector only

import torch
import torch.nn as nn
from torchvision import transforms, models

DATASET_DIR = "dataset"  # root folder for user face images
STREAM_URL = "http://host.docker.internal:8080/video"  # NOTE: If you are running locally, set to 0
MODEL_PATH = "face_model.pth"


class EmbeddingNet(nn.Module):
	"""Match the embedding architecture used in train.py (ResNet50 backbone + embedding layer).

	This reproduces the EmbeddingNet defined in train.py so we can
	load the checkpoint and extract embeddings at inference time.
	"""

	def __init__(self, embed_dim: int, num_classes: int):
		super().__init__()
		backbone = models.resnet50(weights=None)
		in_features = backbone.fc.in_features
		backbone.fc = nn.Identity()
		self.backbone = backbone
		self.embedding = nn.Linear(in_features, embed_dim)
		self.classifier = nn.Linear(embed_dim, num_classes)

	def forward(self, x):
		feats = self.backbone(x)
		emb = self.embedding(feats)
		emb = nn.functional.normalize(emb, p=2, dim=1)
		logits = self.classifier(emb)
		return emb, logits

def ensure_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def capture_user_images(user_name: str, num_images: int = 20) -> None:
	"""Capture images from webcam and save them to a user folder.

	Folder structure: dataset/<user_name>/img_XX.jpg
	"""

	user_dir = os.path.join(DATASET_DIR, user_name)
	ensure_dir(user_dir)

	cap = cv2.VideoCapture(STREAM_URL)
	if not cap.isOpened():
		print("Cannot open webcam")
		return

	print("Press SPACE to capture, ESC to exit early.")

	saved = 0
	while saved < num_images:
		ret, frame = cap.read()
		if not ret:
			print("Failed to grab frame")
			break

		# Show current frame
		cv2.putText(
			frame,
			f"User: {user_name} | Saved: {saved}/{num_images}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 0),
			2,
		)
		cv2.imshow("Capture faces", frame)

		key = cv2.waitKey(1) & 0xFF
		if key == 27:  # ESC
			break
		if key == 32:  # SPACE
			img_path = os.path.join(user_dir, f"img_{saved:02d}.jpg")
			cv2.imwrite(img_path, frame)
			print(f"Saved {img_path}")
			saved += 1

	cap.release()
	cv2.destroyAllWindows()


def load_embedding_model():
	"""Load the trained embedding model and prepare preprocessing."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Train it first.")

	checkpoint = torch.load(MODEL_PATH, map_location=device)
	idx_to_class = checkpoint["idx_to_class"]
	embed_dim = checkpoint.get("embed_dim", 128)
	num_classes = len(idx_to_class)

	model = EmbeddingNet(embed_dim=embed_dim, num_classes=num_classes)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()

	preprocess = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225],
		),
	])

	return model, idx_to_class, device, preprocess, embed_dim


def build_reference_embeddings(model, device, preprocess, embed_dim):
	"""Compute one reference embedding per class from DATASET_DIR.

	This uses the embedding model on all images in dataset/<class>/,
	then averages embeddings per class. Returns (emb_matrix, labels).
	"""
	model.eval()
	embeddings = []
	labels = []

	with torch.no_grad():
		for class_name in sorted(os.listdir(DATASET_DIR)):
			class_dir = os.path.join(DATASET_DIR, class_name)
			if not os.path.isdir(class_dir):
				continue

			class_embs = []
			for fname in os.listdir(class_dir):
				if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
					continue
				img_path = os.path.join(class_dir, fname)
				img_bgr = cv2.imread(img_path)
				if img_bgr is None:
					continue
				img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
				img_pil = transforms.functional.to_pil_image(img_rgb)
				input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
				emb, _ = model(input_tensor)
				class_embs.append(emb.cpu().numpy()[0])

			if class_embs:
				class_embs = np.stack(class_embs, axis=0)
				mean_emb = class_embs.mean(axis=0)
				mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
				embeddings.append(mean_emb)
				labels.append(class_name)

	if not embeddings:
		return None, []

	emb_matrix = np.stack(embeddings, axis=0)  # [num_classes, embed_dim]
	return emb_matrix, labels


def recognize_from_webcam():
	"""Run live recognition using webcam.

	- Draw red rectangle for unknown faces
	- Draw green rectangle with name and probability for known faces
	"""

	try:
		model, idx_to_class, device, preprocess, embed_dim = load_embedding_model()
	except FileNotFoundError as e:
		print(e)
		return

	# Build one reference embedding per identity from the dataset
	ref_embs, ref_labels = build_reference_embeddings(model, device, preprocess, embed_dim)
	if ref_embs is None or not ref_labels:
		print("No reference embeddings found. Ensure dataset is populated.")
		return

	cap = cv2.VideoCapture(STREAM_URL)
	if not cap.isOpened():
		print("Cannot open webcam")
		return

	print("Press ESC to exit.")

	frame_count = 0
	last_detections = []
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Failed to grab frame")
			break
		
		frame_count += 1
		# On most frames, just reuse last detections for smooth display
		if frame_count % 10 != 0 and last_detections:
			for top, right, bottom, left, name, probability, color in last_detections:
				cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
				label = f"{name} {probability * 100:.1f}%"
				cv2.rectangle(frame, (left, bottom + 20), (right, bottom), color, cv2.FILLED)
				cv2.putText(
					frame,
					label,
					(left + 2, bottom + 15),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(255, 255, 255),
					1,
				)
			
			cv2.imshow("Face recognition", frame)
			if cv2.waitKey(1) & 0xFF == 27:
				break
			continue
		
		# Reduce size for faster processing (e.g., half)
		small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

		# Convert frame to RGB for face_recognition
		rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

		# Find all faces in the current frame (locations in resized space)
		locations = face_locations(rgb_frame)

		new_detections = []
		for (top, right, bottom, left) in locations:
			# Crop face from small_frame (BGR)
			face_crop = small_frame[top:bottom, left:right]
			if face_crop.size == 0:
				continue

			# Convert to RGB PIL image for torchvision
			face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
			face_pil = transforms.functional.to_pil_image(face_rgb)

			# Preprocess and run through embedding model
			input_tensor = preprocess(face_pil).unsqueeze(0).to(device)
			with torch.no_grad():
				emb, _ = model(input_tensor)
				emb_np = emb.cpu().numpy()[0]
				emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)

			# Compute cosine similarity to reference embeddings
			sims = ref_embs @ emb_np  # [num_classes]
			best_idx = int(np.argmax(sims))
			best_sim = float(sims[best_idx])

			name = ref_labels[best_idx]
			probability = (best_sim + 1.0) / 2.0  # map cosine [-1,1] -> [0,1]
			# Threshold on similarity to decide if it's "Unknown"
			if best_sim < 0.5:
				display_name = "Unknown"
				color = (0, 0, 255)
			else:
				display_name = name
				color = (0, 255, 0)

			# Scale box back up to original frame size
			top_full = top * 2
			right_full = right * 2
			bottom_full = bottom * 2
			left_full = left * 2

			new_detections.append((top_full, right_full, bottom_full, left_full, display_name, probability, color))

		# Update cache
		last_detections = new_detections

		# Draw current detections
		for top, right, bottom, left, name, probability, color in last_detections:
			cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
			label = f"{name} {probability * 100:.1f}%"
			cv2.rectangle(frame, (left, bottom + 20), (right, bottom), color, cv2.FILLED)
			cv2.putText(
				frame,
				label,
				(left + 2, bottom + 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(255, 255, 255),
				1,
		)

		cv2.imshow("Face recognition", frame)

		key = cv2.waitKey(1) & 0xFF
		if key == 27:  # ESC
			break

	cap.release()
	cv2.destroyAllWindows()


def main():
	print(f"Camera source: {STREAM_URL}")
	print("1 - Capture images for new user")
	print("2 - Run live recognition")
	choice = input("Select mode (1/2): ").strip()

	if choice == "1":
		user_name = input("Enter user name (folder name): ").strip()
		if not user_name:
			print("User name cannot be empty")
			return
		try:
			num = int(input("How many images to capture (default 20): ") or "20")
		except ValueError:
			num = 20
		capture_user_images(user_name, num)
	elif choice == "2":
		recognize_from_webcam()
	else:
		print("Unknown choice")


if __name__ == "__main__":
	main()

