# DRAW Reimplementation with QuickDraw

Reimplementation of [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/pdf/1502.04623.pdf) with the GoogleQuickDraw datasets. 

### Usage

download dataset from [cloud.google/quickdraw_dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false&pageState=("StorageObjectListTable":("f":"%255B%255D")))  (e.g. square.npy, apple.npy or any other) and place it in *data* directory

```
# create and launch virtual environment  
python -m venv pytorch_venv
source pytchorch_venv/bin/activate

# install project dependencies 
pip install -r requirements.txt

# train providing path to the dataset 
python quickdraw_train.py -p data/path_to_dataset.npy

# or generate using previously trained weights 
python utils/generate.py
```

### Example

Apple dataset mixed with square dataset

![AppleSquareQuickDraw](https://raw.githubusercontent.com/bbbrtk/draw-reimplementation/main/image/jablko-kwadrat.gif)

10 datasets mixed together

![DuckFishAppleQuickDraw](https://raw.githubusercontent.com/bbbrtk/draw-reimplementation/main/image/duck-fish-apple.gif)

All dataset containing vehicles mixed together

![VehiclesQuickDraw](https://raw.githubusercontent.com/bbbrtk/draw-reimplementation/main/image/vehicles.gif)

Learning process (3k - 48k steps)

![Learning process](https://raw.githubusercontent.com/bbbrtk/draw-reimplementation/main/image/learning_process.gif)


### Paper

* **Title**: DRAW: A Recurrent Neural Network For Image Generation
* **Authors**: Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, Daan Wierstra
* **Link**: http://arxiv.org/abs/1502.04623
* **Tags**: Neural Network, generative models, recurrent, attention
* **Year**: 2015


### Other sources
  * [TensorFlow implementation](https://github.com/ericjang/draw)
  * [article by Eric Jang](https://blog.evjang.com/2016/06/understanding-and-implementing.html)
  * [QuickDraw data](https://github.com/googlecreativelab/quickdraw-dataset)
  * [DRAW reimplementation](https://github.com/czm0/draw_pytorch)
  * [aleju/papers](https://github.com/aleju/papers/blob/master/neural-nets/DRAW_A_Recurrent_Neural_Network_for_Image_Generation.md)
 
