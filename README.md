# Anime-Face-Extract

  
## Set up environment  
`pip install -r requirement.txt`  

Note that, if you want to utilize the GPU, please follow the instruction [here](https://mxnet.apache.org/versions/1.8.0/get_started?platform=linux&language=python&processor=gpu&environ=pip&).
  

## Face detection  
### reference: [cheese-roll](https://github.com/cheese-roll/light-anime-face-detector)  
<img src="https://user-images.githubusercontent.com/29765855/111080904-0f8b7480-8544-11eb-9cd0-29765ccd6c77.jpg" width="30%" height="30%">  
  

## How to use  

**If you want to extract from a specific data, use -i(input_path) and -o(output_path).**  

EX)  
`python video_detection.py -i .\data\Kaguya\Kaguya1.mp4 -o result/Kaguya/`  

**You just want to extract from all the data, use --processing_all.**  

`python .\video_detection.py --processing_all=True`

**If you want to use GPU --use_gpu**  

`--use-gpu=1` 


## File Structure
#### Before running the code
```
└─ Anime-Face-Extract
   ├─ data
   │  ├─ Anime_title_1
   │  |  ├─ episode_1.mp4
   │  |  ├─ episode_2.mp4
   │  |  ├─ ...
   │  |
   │  ├─ Anime_title_2
   │  |  ├─ episode_1.mp4
   │  |  ├─ episode_2.mp4
   │  |  ├─ ...
   │  |
   │  ├─ ...
   │
   ├─ video_detection.py
   ├─ configs/
   ├─ core/
   ├─ models/
   ├─ requirements.txt
   └─ README.md
```

#### After turning the code
```
└─ Anime-Face-Extract
   ├─ data
   │  ├─ Anime_title_1
   │  |  ├─ episode_1.mp4
   │  |  ├─ episode_2.mp4
   │  |  ├─ ...
   │  |
   │  ├─ Anime_title_2
   │  |  ├─ episode_1.mp4
   │  |  ├─ episode_2.mp4
   │  |  ├─ ...
   │  |
   │  ├─ ...
   │
   ├─ result
   │  ├─ Anime_title_1
   │  |  ├─ 0.jpg
   │  |  ├─ 1.jpg
   │  |  ├─ ...
   │  |  ├─ ROI.csv
   │  |
   │  ├─ Anime_title_2
   │  |  ├─ 0.jpg
   │  |  ├─ 1.jpg
   │  |  ├─ ...
   │  |  ├─ ROI.csv   
   │  |
   │  ├─ ...
   │
   ├─ video_detection.py
   ├─ configs/
   ├─ core/
   ├─ models/
   ├─ requirements.txt
   └─ README.md
```
  

<!--
## Developers  
|Name|Github|Email|
|:--:|:--:|:--:|
|김진호|[@kimjinho1](https://github.com/kimjinho1)|rlawlsgh8113@naver.com|
|남영우|[@yw0nam](https://github.com/yw0nam)|spow2544@gmail.com|
|이경현|[@Jovinus](https://github.com/Jovinus)|lkh256@gmail.com|
-->
