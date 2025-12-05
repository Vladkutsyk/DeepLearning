import os
import cv2
import numpy as np
from face_recognition import load_image_file, face_encodings, face_locations, face_distance

DATASET_DIR = "dataset"  # root folder for user face images
STREAM_URL = "http://host.docker.internal:8080/video" # NOTE: If you are running locally, simply change it to 0. STREAM_URL = 0

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


def load_known_faces():
	"""Load all faces from DATASET_DIR.

	Expects directory structure dataset/<user_name>/*.jpg
	Returns lists: encodings, names.
	"""

	known_encodings = []
	known_names = []

	if not os.path.exists(DATASET_DIR):
		return known_encodings, known_names

	for user_name in os.listdir(DATASET_DIR):
		user_dir = os.path.join(DATASET_DIR, user_name)
		if not os.path.isdir(user_dir):
			continue

		for file_name in os.listdir(user_dir):
			if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
				continue
			img_path = os.path.join(user_dir, file_name)

			image = load_image_file(img_path)
			encodings = face_encodings(image)
			if not encodings:
				continue

			known_encodings.append(encodings[0])
			known_names.append(user_name)

	return known_encodings, known_names


def recognize_from_webcam():
	"""Run live recognition using webcam.

	- Draw red rectangle for unknown faces
	- Draw green rectangle with name and probability for known faces
	"""

	known_encodings, known_names = load_known_faces()
	if not known_encodings:
		print("No known faces in dataset. First run capture mode.")
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

		# Find all faces and their encodings in the current frame
		locations = face_locations(rgb_frame)
		encodings = face_encodings(rgb_frame, locations)

		new_detections = []
		for (top, right, bottom, left), face_encoding in zip(locations, encodings):
			# Compare with known faces
			distances = face_distance(known_encodings, face_encoding)

			if len(distances) > 0:
				best_idx = np.argmin(distances)
				best_distance = distances[best_idx]
				# Convert distance to a rough "similarity" probability
				probability = float(max(0.0, 1.0 - best_distance))

				match = best_distance < 0.6  # typical threshold
				name = known_names[best_idx] if match else "Unknown"
				color = (0, 255, 0) if match else (0, 0, 255)  # green or red
			else:
				probability = 0.0
				name = "Unknown"
				color = (0, 0, 255)

			top *= 2
			right *= 2
			bottom *= 2
			left *= 2

			new_detections.append((top, right, bottom, left, name, probability, color))

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

