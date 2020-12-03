### DRAW Reimplementation with QuickDraw

Reimplementation of [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/pdf/1502.04623.pdf) with the GoogleQuickDraw datasets. 

### Usage

0. create virtual environment  

 `python -m venv pytorch_venv`

1. install project dependencies 

 `pip install -r requirements.txt`

2. download dataset from [cloud.google/quickdraw_dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false&pageState=("StorageObjectListTable":("f":"%255B%255D")))  (e.g. square.npy or apple.npy) and place it in *data* directory

3. train providing path to the dataset 

`python quickdraw_train.py -p data/path_to_dataset.npy`

4. or generate using previously trained weights 

`python utils/generate.py`


### Paper

* **Title**: DRAW: A Recurrent Neural Network For Image Generation
* **Authors**: Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, Daan Wierstra
* **Link**: http://arxiv.org/abs/1502.04623
* **Tags**: Neural Network, generative models, recurrent, attention
* **Year**: 2015


### Other
  * Based on implementation [czm0/draw_pytorch](https://github.com/czm0/draw_pytorch)
  * [article by Eric Jang](https://blog.evjang.com/2016/06/understanding-and-implementing.html)
  * [QuickDraw data](https://github.com/googlecreativelab/quickdraw-dataset)
