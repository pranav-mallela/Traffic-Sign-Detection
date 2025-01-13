# Traffic Sign Detection using PyTorch

What it is: A simple image classifier - classifies into four categories, traffic light, stop sign, speed limit sign, and crosswalk. 

Link to dataset: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection 

Model weights are in `./website/classifier.pt`

How to test the model:

1. The website is not hosted yet. Open `index.html` in a browser.
2. Run `python app.py` in your terminal.
3. Press the *Activate Camera* button on the website.
4. Point the camera to a traffic sign to see the classification.

If you want to retrain it: 

1. Upload `traffic_sign_detection_colab_dl.ipynb` to Google Colab.
2. Download data from the data source and upload to Google Drive.
3. Run the cells.
