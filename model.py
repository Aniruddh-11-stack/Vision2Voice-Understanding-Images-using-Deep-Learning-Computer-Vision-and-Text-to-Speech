import numpy as np
import cv2
import os
import shutil
from pickle import load
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from ultralytics import YOLO
from PIL import Image

class Vision2VoicePredictor:
    def __init__(self, models_dir):
        """
        Initializes the model architecture by loading weights from the backend
        models directory instead of Google Drive paths.
        """
        self.models_dir = models_dir
        
        # Load VGG16 base model for feature extraction
        base_model = VGG16(include_top=True)
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        
        # Paths to required weights - Expecting these to exist in models_dir
        tokenizer_path = os.path.join(models_dir, 'caption_train_tokenizer.pkl')
        model_path = os.path.join(models_dir, 'modelConcat_1_89.h5')
        
        # Check if they exist
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            self.ready = False
        else:
            self.ready = True
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = load(f)
            self.model = load_model(model_path)
            
        # YOLO model for object cropping
        # By default this will download to the running directory if missing
        self.yolo = YOLO("yolov8n.pt") 
        
        self.max_length = 33
        self.vocab_size = 7506
        self.beam_width = 10

    def extract_feature(self, np_img):
        """Extracts features from an image using VGG16."""
        # VGG16 expects 224x224 RGB
        img_resized = cv2.resize(np_img, (224, 224))
        x = img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.feature_extractor.predict(x, verbose=0)
        return features

    def _generate_caption_beam(self, photo):
        """Generates captions using beam search."""
        sequence = self.tokenizer.texts_to_sequences(['<START>'])[0]
        sequence = pad_sequences([sequence], maxlen=self.max_length)
        
        # First prediction
        model_softMax_output = np.squeeze(self.model.predict([photo, sequence], verbose=0))
        most_likely_seq = np.argsort(model_softMax_output)[-self.beam_width:]
        most_likely_prob = np.log(model_softMax_output[most_likely_seq])

        most_likely_cap = [[[self.tokenizer.index_word[word_idx]]] for word_idx in most_likely_seq]

        for _ in range(self.max_length):
            temp_prob = np.zeros((self.beam_width, self.vocab_size))
            for j in range(self.beam_width):
                if most_likely_cap[j][-1] != ['end']:
                    num_words = len(most_likely_cap[j])
                    seq = self.tokenizer.texts_to_sequences(most_likely_cap[j])
                    seq = pad_sequences(np.transpose(seq), maxlen=self.max_length)
                    model_softMax_output = self.model.predict([photo, seq], verbose=0)
                    temp_prob[j,] = (1 / num_words) * (most_likely_prob[j] * (num_words - 1) + np.log(model_softMax_output))
                else:
                    temp_prob[j,] = most_likely_prob[j] + np.zeros(self.vocab_size) - np.inf
                    temp_prob[j, 0] = most_likely_prob[j]

            x_idx, y_idx = np.unravel_index(temp_prob.flatten().argsort()[-self.beam_width:], temp_prob.shape)

            most_likely_cap_temp = []
            for j in range(self.beam_width):
                most_likely_prob[j] = temp_prob[x_idx[j], y_idx[j]]
                cap_copy = most_likely_cap[x_idx[j]].copy()
                if cap_copy[-1] != ['end']:
                    cap_copy.append([self.tokenizer.index_word[y_idx[j]]])
                most_likely_cap_temp.append(cap_copy)

            most_likely_cap = most_likely_cap_temp.copy()

            finished = all(cap[-1] == ['end'] for cap in most_likely_cap)
            if finished:
                break

        # Flatten nested structures
        final_caption = []
        for j in range(self.beam_width):
            # Flatten lists of single element lists
            flat_cap = [item[0] for item in most_likely_cap[j][:-1]] # skip "end" if present
            final_caption.append(' '.join(flat_cap))
            
        return final_caption, most_likely_prob

    def generate_best_caption(self, np_img):
        """Generates the single best string caption for the given image numpy array (RGB format)."""
        photo = self.extract_feature(np_img)
        captions, probs = self._generate_caption_beam(photo)
        
        if not captions:
            return ""
            
        result_dict = dict(zip(probs, captions))
        max_prob = max(result_dict.keys())
        best_caption = result_dict[max_prob]
        return best_caption

    def analyze_full_image(self, image_path):
        """
        The Vision2Voice method logic:
        1. Parse the overall image.
        2. Use YOLO to crop entities.
        3. Parse the cropped entities.
        4. Join the captions.
        """
        if not self.ready:
            raise RuntimeError("Model weight files were not found in the 'models' directory.")
            
        # 1. Base Image Caption
        base_img = cv2.imread(image_path)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        base_caption = self.generate_best_caption(base_img)
        
        # 2. YOLO Crop Extractor
        results = self.yolo.predict(image_path, verbose=False)
        result = results[0]
        
        cropped_captions = []
        pil_image = Image.open(image_path)
        
        for box in result.boxes:
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            x_min, y_min, x_max, y_max = cords
            
            cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
            # Convert to numpy and then to RGB (cv2 style formatting expectation)
            np_crop = np.array(cropped_image)
            if len(np_crop.shape) == 2:
                np_crop = cv2.cvtColor(np_crop, cv2.COLOR_GRAY2RGB)
            elif np_crop.shape[2] == 4:
                np_crop = cv2.cvtColor(np_crop, cv2.COLOR_RGBA2RGB)
                
            crop_caption = self.generate_best_caption(np_crop)
            cropped_captions.append(crop_caption)
            
        # 3 & 4. Returning Data
        all_captions = [base_caption] + cropped_captions
        collective_caption = " ".join(all_captions)
        
        return base_caption, collective_caption
