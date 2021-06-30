Team: Jelena Mrdak  

## :package: Installation
:exclamation: Requirements: OpenCV, Keras, cmake

For trainging with GPU (Manjaro): cuda, python-cuda, python-pycuda, cudnn, tensorflow-cuda, python-tensorflow-cuda, pydot

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
├── dcgan  
├── demo  
│   ├── one_letter  
│   │   ├── pogresno_prediktovana_slova.pdf  
│   │   ├── test_podaci_random_25.pdf  
│   │   └── trening_podaci.pdf  
│   ├── one_two  
│   │   ├── pogresno_prediktovana_slova.pdf  
│   │   ├── test_podaci_random_25.pdf  
│   │   └── trening_podaci.pdf  
│   ├── two_letters  
│   │   ├── first  
│   │   │   ├── pogresno_prediktovana_slova.pdf  
│   │   │   ├── test_podaci_random_25.pdf  
│   │   │   └── trening_podaci.pdf  
│   │   └── second  
│   │       ├── pogresno_prediktovana_slova.pdf  
│   │       ├── test_podaci_random_25.pdf  
│   │       └── trening_podaci.pdf  
│   └── demo.ipynb  
├── include  
│   ├── image.cpp  
│   └── image.h   
├── merge  
│   ├── CMakeLists.txt  
│   └── main.cpp  
├── model   
│   ├── one_letter  
│   │   └── model.h5  
│   ├── one_two  
│   │   └── model.h5  
│   └── two_letters  
│       ├── first  
│       │   └── model.h5  
│       └── second  
│           └── model.h5  
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
├── classification_report.py  
├── model.pdf  
├── model.py  
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

