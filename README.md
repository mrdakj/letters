## :package: Installation
:exclamation: Requirements: OpenCV, Keras, cmake

For trainging with GPU (Manjaro): cuda, python-cuda, python-pycuda, cudnn, tensorflow-cuda, python-tensorflow-cuda

## Structure

```
letters
├── components  
│   ├── CMakeLists.txt  
│   └── main.cpp  
├── dataset  
│   ├── train  
│   │   └── one_letter  
│   │       ├── bold  
│   │       ├── medium  
│   │       └── normal 
│   ├── validation  
│   │   └── one_letter  
│   │       ├── bold  
│   │       ├── medium  
│   │       └── normal  
│   └── test  
│       └── one_letter  
│           ├── bold  
│           ├── medium  
│           └── normal  
├── include  
│   ├── image.cpp  
│   └── image.h   
├── merge  
│   ├── CMakeLists.txt  
│   └── main.cpp  
├── model   
│   ├── one_letter  
│   └── two_letters  
├── report  
│   ├── one_letter  
│   │   ├── accuracy.pdf  
│   │   ├── confusion_matrix.pdf  
│   │   ├── loss.pdf  
│   │   └── report.pdf  
│   ├── one_two  
│   │   ├── accuracy.pdf  
│   │   ├── confusion_matrix.pdf  
│   │   ├── loss.pdf  
│   │   └── report.pdf  
│   └── two_letters  
│       ├── first  
│       │   ├── accuracy.pdf  
│       │   ├── confusion_matrix.pdf  
│       │   ├── loss.pdf  
│       │   └── report.pdf  
│       └── second  
│           ├── accuracy.pdf  
│           ├── confusion_matrix.pdf  
│           ├── loss.pdf  
│           └── report.pdf  
├── network.py  
├── network_one_two.py  
└── README.md   

```

## Setup and Usage

1. Download dataset

2. Create components - generate letter images
    ```sh
    cd components 
    mkdir build && cd build
    cmake .. && cmake --build .
    ./main ../../dataset/train/one_letter prepare
    ./main ../../dataset/validation/one_letter prepare
    ./main ../../dataset/test/one_letter prepare

    ```

3. Merge two letters
    ```sh
    cd merge 
    mkdir build && cd build
    cmake .. && cmake --build .
    ./main train first
    ./main validation first
    ./main test first
    ./main train second
    ./main validation second
    ./main test second

    ```

4. Train networks
    ```sh
    # one letter
    python3 network.py one
    # two letters
    python3 network.py first
    python3 network.py second
    # train one vs two network
    python3 network_one_two.py

    ```

## Results

### One letter

Training size: 514860  
Test size: 58677  

Train best val accuracy model: accuracy: 99.86%   
Test best val accuracy model: accuracy: 98.48%  

### Two letters glued together - fst letter recognition  

training: 200 per image pair, no rotations  
validation: 40 per image pair, no rotations  

Train size: 405600 (200x26x26x3)  
Test size: 81120 (40x26x26x3)  

Train best val accuracy model: accuracy: 99.74%  
Test best val accuracy model: accuracy: 96.91%  

My handwritings:  
a 0.9526525955504849  
b 0.9907578558225508   
c 0.9116766467065869   
d 0.983453981385729   
e 0.9644268774703557   
f 0.9713804713804713   
g 0.9904420549581839   
h 0.9915966386554622   
k 0.989010989010989    
l 0.9835680751173709   
m 0.9908361970217641   
n 0.9778368794326241   

### One vs two letters  
Train best val accuracy model: accuracy: 99.56%  
Test best val accuracy model: accuracy: 99.23%  
