
import numpy as np
import cv2, os, audio
import requests
from tqdm import tqdm
import torch, face_detection
from models import Wav2Lip
from moviepy.editor import VideoFileClip, AudioFileClip


class LipSyncConfig:
    def __init__(self):
        self.face = None
        self.audio = None
        self.outfile = 'results/result_voice.mp4'
        self.static = False
        self.fps = 25.0
        self.pads = [0, 10, 0, 0]
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 128
        self.resize_factor = 1
        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.rotate = False
        self.nosmooth = False
        self.img_size = 96




def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	global config
	global device
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False, device=device)

	batch_size = config.face_det_batch_size

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = config.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)

		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not config.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results

def datagen(frames, mels):
	global config
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if config.box[0] == -1:
		if not config.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = config.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if config.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (config.img_size, config.img_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= config.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, config.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, config.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch




def _load(checkpoint_path):
	global device
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def generate_video(face_link,audio_link,checkpoint_path):

	global config
	config = LipSyncConfig()
	# "checkpoints/wav2lip_gan.pth"
	mel_step_size = 16
	global device
	device = 'cuda' if torch.cuda.is_available() else 'cpu'



	if "http" in face_link:
		response = requests.get(face_link)
		image_data = response.content
		content_type = response.headers.get('content-type')
		if content_type:
			extension=content_type.split("/")[-1]
		else:
			extension='jpg'
		video_file_name=f"sample_data/face_image.{extension}"
		with open (video_file_name,'wb') as f:
			f.write(image_data)
		print("FILENAME=",video_file_name)

	if "http" in audio_link:
		response=requests.get(audio_link)
		audio_data=response.content
		content_type=response.headers.get('content-type')
		if content_type:
			extension=content_type.split("/")[-1]
		else:
			extension='mp3'
		audio_file_name=f"sample_data/audio.{extension}"
		with open(audio_file_name,'wb') as f:
			f.write(audio_data)

	config.face=video_file_name
	config.audio=audio_file_name
	if not os.path.isfile(config.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif config.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(config.face)]
		fps = config.fps
		print("config=",config)
		print("")

	# if provided file is not .wav
	if not config.audio.endswith('.wav'):
		print('Extracting raw audio...')
		# Load the audio clip
		audio_clip = AudioFileClip(config.audio)
		# Define the path for the temporary WAV file
		temp_wav_path = 'temp/temp.wav'
		# Write the audio clip to the temporary WAV file
		audio_clip.write_audiofile(temp_wav_path)
		# Close the audio clip
		audio_clip.close()
		config.audio = temp_wav_path

	wav = audio.load_wav(config.audio, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = config.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi',
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	result_video_path = 'temp/result.avi'
	output_video_path = config.outfile

	# Load the audio and video clips
	audio_clip = AudioFileClip(config.audio)
	video_clip = VideoFileClip(result_video_path)

	# Set the audio of the video clip to the extracted audio
	video_clip = video_clip.set_audio(audio_clip)

	# Write the final video with merged audio
	video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

	# Close the clips
	audio_clip.close()
	video_clip.close()

	try:
		if os.path.isfile(result_video_path):
			os.remove(result_video_path)
	except:
		pass

	return output_video_path
