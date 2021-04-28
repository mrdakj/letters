## :package: Installation
:exclamation: Requirements: OpenCV, Keras, cmake

For trainging with GPU (Manjaro): cuda, python-cuda, python-pycuda, cudnn, tensorflow-cuda, python-tensorflow-cuda

## Structure

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
One letter

Training size: 514860
Test size: 58677
Train: accuracy: 99.82%
Test: accuracy: 98.44%

test network one letter
a 0.9753566796368353
b 0.9641693811074918
c 0.9860082304526749
d 0.9964539007092199
e 0.9960915689558906
f 0.987045341305431
g 0.9762731481481481
h 0.9937810945273632
i 0.9956204379562044
j 0.9906890130353817
k 0.9947665056360708
l 0.9677312042581504
m 0.9924098671726755
n 0.9878787878787879
o 0.9889860706187237
p 0.9717314487632509
q 0.9927623642943305
r 0.9632183908045977
s 0.9929519071310116
t 0.976629766297663
u 0.9711029711029711
v 0.9917081260364843
w 0.9974160206718347
x 0.9771573604060914
y 0.9986850756081526
z 0.9898063200815495
0.9836399750107973

Two letters

training: 200 per image pair, no rotations
validation: 40 per image pair, no rotation

Train size: 405600 (200x26x26x3)
Test size: 81120 (40x26x26x3)

Train: accuracy: 98.69%
Test: accuracy: 96.56%

a 0.9538461538461539
b 0.9538461538461539
c 0.9458333333333333
d 0.9314102564102564
e 0.992948717948718
f 0.9852564102564103
g 0.9846153846153847
h 0.9775641025641025
i 0.9560897435897436
j 0.978525641025641
k 0.9887820512820513
l 0.9426282051282051
m 0.969551282051282
n 0.9621794871794872
o 0.926923076923077
p 0.9419871794871795
q 0.9637820512820513
r 0.969551282051282
s 0.9814102564102564
t 0.9657051282051282
u 0.9564102564102565
v 0.9682692307692308
w 0.9634615384615385
x 0.9721153846153846
y 0.9759615384615384
z 0.9961538461538462

