# MURA-Model
A model for the MURA dataset

## Usage
To evaluate the model, run the following command:
```bash
python3 evaluation.py ./MURA-v1.0

Usage: evaluation.py [options] data_dir

Options:
  -h, --help            show this help message and exit
  -p PHASE, --phase=PHASE
                        valid or train, assume the same directory structure
                        with the training set, where there should be a file
                        named <PHASE>.csv in the data directory
  -m MODEL, --model=MODEL
                        the model to evaluate, one of [densenet161,
                        densenet169, resnet50, vgg19, agnet]
  -s STUDY, --study=STUDY
                        for evaluating on a specific study
  -d, --draw            draw roc curves
```

To localize the disease, run the following command:
```bash
python3 localize.py <path to the image>
```

To retrieve the similar images, run the following command:
```bash
python3 similar.py <path_to_the_img> <path_to_the_dataset>

Usage: similar.py [option] img_path dir_path

Options:
  -h, --help  show this help message and exit
  -d, --draw  show the images
```
