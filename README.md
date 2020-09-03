
# Multimodal Emotion Recognition base model for MERC2020
	- This code is based on keras(2.2.4) and tensorflow(1.14).
	- This is only for reference and participants are not required to use it.

## Installation
	```
	pip3 install -r requirement.txt
	```

## Text base model
	- Root path
	```
	cd modules/text
	```

### Usage examples
	1. Load dataset
	```
	python3 load.py
	```
	2. Train model
	```
	CUDA_VISIBLE_DEVICES=0 python3 train.py
	```

### Performance
	- Accuracy: 45.23% (validation dataset)


## Video base model
	- Root path
	```
	cd modules/video
	```

### Usage examples
	1. Get features using VGG face
	```
	CUDA_VISIBLE_DEVICES=0 python3 video_feature_extract.py
	```
	2. Load dataset
	```
	python3 load.py
	```
	3. Train model
	```
	CUDA_VISIBLE_DEVICES=0 python3 train.py
	```

### Performance
 	- Accuracy: 29.71% (valiation dataset)


## Multimodal base model
	- Root path
	```
	cd modules/multimodal
	```
### Usage examples
	1. Get features using VGG face
	```
	CUDA_VISIBLE_DEVICES=0 python3 video_feature_extract.py
	```
	2. Load dataset
	```
	python3 load.py
	```
	3. Train model
	```
	CUDA_VISIBLE_DEVICES=0 python3 train.py
	```

### Performance
 	- Accuracy: 29.71% (valiation dataset)



### Installation
	-
