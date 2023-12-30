# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Part of this [Starter project](https://github.com/udacity/aipnd-proj)

## our data directory (args.data_directory) contains the `test` `train` and `valid` subdirectories:
```
- data_directory
  - train
    - class_
      - image1.jpg
      - image2.jpg
      ...
    - class_2
      - image1.jpg
      - image2.jpg
      ...
    ...
  - valid
    - class_1
      - image1.jpg
      - image2.jpg
      ...
    - class_2
      - image1.jpg
      - image2.jpg
      ...
    ...
```


## To use these scripts, you would run them from the command line, providing necessary arguments. For example:
```
python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 256 --epochs 10 --gpu

```

This command trains a model on the 'flowers' dataset, saves checkpoints in the 'checkpoints' directory, uses the VGG16 architecture, a learning rate of 0.001, 256 hidden units in the classifier, runs for 10 epochs, and utilizes GPU acceleration if available.

# Predicting on a directory:
```
python predict.py --data_directory flowers --checkpoint checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```
# Predicting on a single image:
```

python predict.py --image flowers/valid/class_1/image_00005.jpg --checkpoint checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu

```

