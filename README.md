
# Multimodal Emotion Recognition baseline code for MERC2020
	- This is baseline code for MERC2020
	- This code is based on keras(2.2.4) and tensorflow(1.14).
	- You need to prepare multimodal emotion dataset and speech recognition baseline code.
	- Useful codes for data preperation are uploaded in '/utils' (10/23)

## Installation
	pip3 install -r requirement.txt


## Useful code for data preperation

	mp4 to jpg: utils/video_to_image.py
	text emb to npy: utils/text_to_npy.py
	label to npy: utils/label_to.npy.py

## Text baseline model

### Root path
	cd modules/text

### Usage examples

	1. Load dataset
	python3 load.py

	2. Train model
	CUDA_VISIBLE_DEVICES=0 python3 train.py

### Performance
	Accuracy: 45.23% (validation dataset)


## Video baseline model

### Root path
	cd modules/video

### Usage examples

	1. Get features using VGG face
	CUDA_VISIBLE_DEVICES=0 python3 video_feature_extract.py

	2. Load dataset
	python3 load.py

	3. Train model
	CUDA_VISIBLE_DEVICES=0 python3 train.py

### Performance
 	- Accuracy: 29.71% (valiation dataset)


## Multimodal baseline model

### Root path
	cd modules/multimodal

### Usage examples
	1. Get bottleneck features from each single modal
	CUDA_VISIBLE_DEVICES=0 python3 feature_extract.py
	
	2. Train model
	CUDA_VISIBLE_DEVICES=0 python3 train.py
	
### Performance
 	- Accuracy: 52.56% (valiation dataset)

### License
	- The source code, pre-trained models, datasets are available under MERC2020 competition.
