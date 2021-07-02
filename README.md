Team: Jelena Mrdak  

## :package: Installation
:exclamation: Requirements: OpenCV, Keras, cmake

For trainging with GPU (Manjaro): cuda, python-cuda, python-pycuda, cudnn, tensorflow-cuda, python-tensorflow-cuda, pydot

## Structure

```
letters
├── components                                              <---  create n smaller images of a letter from the bigger image with n letters  
│   ├── CMakeLists.txt  
│   └── main.cpp  
├── dataset                                                 <---  dataset containing letters of english alphabet  
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
├── dcgan                                                   <---  network for generating new letters based on dataset  
│   ├── report  
│   └── dcgan.py  
├── demo                                                    <---  jupyther notebook used to demonstrate models  
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
├── include                                                <---  helper class for image manipulation  
│   ├── image.cpp  
│   └── image.h   
├── merge                                                  <---  used to generate dataset for bigrams  
│   ├── CMakeLists.txt  
│   └── main.cpp  
├── model                                                  <---  trained models for letters recognition  
│   ├── one_letter  
│   │   └── model.h5  
│   ├── one_two  
│   │   └── model.h5  
│   └── two_letters  
│       ├── first  
│       │   └── model.h5  
│       └── second  
│           └── model.h5  
├── report                                                 <---  model reports (data distribution, curves, scores, classification reports, confusion matrix)  
│   ├── one_letter  
│   │   ├── accuracy.pdf  
│   │   ├── confusion_matrix_test.pdf  
│   │   ├── confusion_matrix_val.pdf  
│   │   ├── loss.pdf  
│   │   ├── report_test.pdf  
│   │   ├── report_val.pdf  
│   │   ├── scores.txt  
│   │   ├── test_podaci_raspodela.pdf  
│   │   ├── trening_podaci_raspodela.pdf  
│   │   └── validacioni_podaci_raspodela.pdf  
│   ├── one_two  
│   │   ├── accuracy.pdf  
│   │   ├── confusion_matrix_test.pdf  
│   │   ├── confusion_matrix_val.pdf  
│   │   ├── loss.pdf  
│   │   ├── report_test.pdf  
│   │   ├── report_val.pdf  
│   │   ├── scores.txt  
│   │   ├── test_podaci_raspodela.pdf  
│   │   ├── trening_podaci_raspodela.pdf  
│   │   └── validacioni_podaci_raspodela.pdf  
│   └── two_letters  
│       ├── first  
│       │   ├── accuracy.pdf  
│       │   ├── confusion_matrix_test.pdf  
│       │   ├── confusion_matrix_val.pdf  
│       │   ├── loss.pdf  
│       │   ├── report_test.pdf  
│       │   ├── report_val.pdf  
│       │   ├── scores.txt  
│       │   ├── test_podaci_raspodela.pdf  
│       │   ├── trening_podaci_raspodela.pdf  
│       │   └── validacioni_podaci_raspodela.pdf  
│       └── second  
│           ├── accuracy.pdf  
│           ├── confusion_matrix_test.pdf  
│           ├── confusion_matrix_val.pdf  
│           ├── loss.pdf  
│           ├── report_test.pdf  
│           ├── report_val.pdf  
│           ├── scores.txt  
│           ├── test_podaci_raspodela.pdf  
│           ├── trening_podaci_raspodela.pdf  
│           └── validacioni_podaci_raspodela.pdf  
├── classification_report.py                               <---  helper class for creating reports  
├── model.pdf  
├── model.py                                               <---  class defining model and its helper methods  
├── network.py                                             <---  network used to train model for one and two letters recognition  
├── network_one_two.py                                     <---  network used to train model for one vs. two letters classification  
├── README.md   
└── test_dcgan.py                                          <---  generate report for gan images  

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

5. To train DCGAN
    ```sh
    python3 dcgan.py [letter] train
    # example
    python3 dcgan.py a train

    ```

5. To use DCGAN
    ```sh
    python3 dcgan.py [letter] generate
    # example
    python3 dcgan.py a generate

    ```
