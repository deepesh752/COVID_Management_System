The purpose of this study was to build a web application that took advantage of existing and easily available infrastructure, i.e. X-ray machines and CT-Scans for quick and efficient detection of COVID-19, along with providing other necessary details to the user. For the efficient detection of COVID-19 we propose the usage of SRCNN algorithm for increasing the resolution of the image followed by a classification algorithm for COVID-19 detection. The following key phases were considered while developing the COVID-19 detection model: Image pre-processing, Image degradation, Image superresolution, and CNN. Secondary information that was selected to be displayed in the web app included recent statistics on COVID-19 with graphical representation ,and general health information addressing the same.

The SRCNN super resolution model is required to learn the mapping between low resolution images to high resolution images. To achieve this Set5 and Set14 datasets were used. These are common evaluation dataset for Super Resolution of images, containing various images from buildings to animal faces. Images were pre-processed and the dimensions were converted into 224x224x3  For training(750 images) and testing(150 images) purposes  covid-chestxray-dataset dataset is used which contains the X-Rays of lungs categorized into two classes namely COVID and NORMAL Hyper tuning of the parameter was done with binary cross-entropy and ADAM optimizer per epoch for 20 epochs.

![image](https://user-images.githubusercontent.com/39914367/125448925-0ec6c84d-269b-4025-9b62-3cce5c80d195.png)

Information sources used to supplement secondary information, i.e. cases and health information were taken from two different sources. Covid statistics were taken from COVID-19 API which was independently built by Kyle Redelinghuys. The data is sourced from COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University. The data present is automatically updated on a regular(hourly to daily) basis. General health information query related to COVID-19 is hard coded and taken from the World Health Organization website.

Flask is a micro-framework which was chosen to supplement the project, reason being its simplicity, ease of building prototypes and smaller codebase(leading in reduced application size), along with ease of integration with machine learning models.

![image](https://user-images.githubusercontent.com/39914367/125448780-a5f61e96-c58d-4870-8cf2-8d6203d80dcb.png)


##Results

The accuracy that was attained after 20 epochs of training was 95%. This accuracy was higher than the accuracy of the model in which image super-resolution was not considered. Due to image super-resolution, our model achieved this accuracy. For hyper tuning, we hyper-tuned loss function, optimizer, epochs, and the learning rate. The corresponding values for the hyperparameters were: binary cross entropy as the loss function, ADAM as the optimizer, 20 is the number of epochs, and 0.002 as the learning rate. 

![image](https://user-images.githubusercontent.com/39914367/125449716-2aef0c01-caf4-440d-8710-6ccb2ffeec46.png)
Fig. Accuracy for training and testing model

![image](https://user-images.githubusercontent.com/39914367/125449616-eaec38cf-744f-4478-876f-a8da75803a83.png)
Fig. Loss for training and testing model  

One can come to a realization that the loss is decreasing gradually till 15-17 epochs and after that, the graph is pretty much stagnant. This was the reason behind “why 20 epochs?”.   Here the loss we achieved was ~18%. 
Therefore, from the results, we can say that due to the SRCNN algorithm, the classification was better of whether or not a patient has COVID-19. 
