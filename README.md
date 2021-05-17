## :package: Installation
:exclamation: Requirements: OpenCV, Keras, cmake

For trainging with GPU (Manjaro): cuda, python-cuda, python-pycuda, cudnn, tensorflow-cuda, python-tensorflow-cuda

## Structure

```
letters_recognition  
├── components  
│   ├── CMakeLists.txt  
│   └── main.cpp  
├── dataset  
│   ├── train  
│   │   └── one_letter  
│   │       ├── bold  
│   │       ├── medium  
│   │       └── normal  
│   └── validation  
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
├── one   
│   ├── CMakeLists.txt  
│   └── main.cpp  
├── models_results.txt  
├── network_one_letter.py      
├── network_two_letters.py     
├── README.md   
├── test_network_one_letter.py  
└── test_network_two_letters.py  

```

## Setup and Usage

1. Create model dir
    ```sh
    mkdir -p model/one_letter
    mkdir -p model/two_letters

    ```
2. Download dataset

3. Create components - generate letter images
    ```sh
    cd components 
    mkdir build && cd build
    cmake .. && cmake --build .
    ./main ../../dataset/train/one_letter rotate  
    ./main ../../dataset/validation/one_letter rotate  

    ```

4. Prepare data - resize and rotate
    ```sh
    cd one 
    mkdir build && cd build
    cmake .. && cmake --build .
    ./main train
    ./main validation

    ```

5. Merge two letters
    ```sh
    cd merge 
    mkdir build && cd build
    cmake .. && cmake --build .
    ./main

    ```

6. Train networks
    ```sh
    # one letter
    python3 network_one_letter.py 
    # two letters
    python3 network_two_letters.py 

    ```

7. Test networks
    ```sh
    # one letter
    python3 test_network_one_letter.py
    # two letters
    python3 test_network_two_letters.py 

    ```

## Results

### One letter

Training size: 514860  
Test size: 58677  

Train best val accuracy model: accuracy: 99.85%  
Test best val accuracy model: accuracy: 98.44% (best so far 98.48)  

### Two letters glued together - fst letter recognition  

training: 200 per image pair, no rotations  
validation: 40 per image pair, no rotations  

Train size: 405600 (200x26x26x3)  
Test size: 81120 (40x26x26x3)  

Train best val accuracy model: accuracy: 99.60%  
Test best val accuracy model: accuracy: 96.83%  

my handwritting recall   
a 0.9515116942384484  
b 0.9741219963031423  
c 0.9311377245508982  
d 0.9824198552223371  
e 0.9654150197628458  
f 0.9722222222222222  
g 0.997610513739546   
h 0.9859943977591037  
k 0.9914529914529915  
l 0.9906103286384976  
m 0.9885452462772051  
n 0.9769503546099291  
