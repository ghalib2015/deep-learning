# deep-learning OT1 project
This is the github repository of deep learning OT1 project. </br>
It contains the models for the 3 tasks. </br>
- The architectures are defined on classifier.py with 3 models, one for each task. </br>
  - colorClassifier</br>
  - BoxClassifier</br>
  - SeqClassifier</br>
- The training was done with:</br>
  - training.py for color</br>
  - training2.py for box</br>
  - training3.py for sequence</br>
- The evolution of the models can be visualized using tensorboard on runs directory, one for each model</br>
- The models are stored on .pth files, one for each model
- The box and sequence models can be tested using respectively testing.py and testing2.py, there is also prediction.zip which contain all the predicted bounding box for validation set

More information can be found on the Drive report from this link: https://docs.google.com/document/d/1-MjN0XJE2BftC_5SEeBTJ2u4SchuFOgHryqnokh4GCg/edit?usp=sharing
