# Chinese Calligraphy Detection

## Introduction

In order to detect the chinese calligraphy text, we can deal with this task step by step.

1. First, use CTPN model to train the rpn and get the geographical positions of each line in the chinese calligraphy image.
2. Then, predict the sub-image that contains just one single line of the overall text with the trained model,and collect all of them.
3. On top of that, recognize the chinese character in these sub-images

Therefore,the two main models are：

1. CTPN for location detection
2. Densenet + ctc for character recognition

## Getting Started

### system environment

OS: win10  
Python: 3.6  
tensorflow:1.9.0  
keras:2.2.4  
