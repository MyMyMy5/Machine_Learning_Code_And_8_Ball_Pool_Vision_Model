<Context>
The game I am talking about is called 8 Ball Pool. 
My task was to create a machine learning model that could extend the small white cue line that is going from the center of the cue-ball and out. In more specific terms, when you aim towards a cue-ball, there is a small white line that goes out from the center of a cue-ball. That specific line is the direction in which the cue-ball will go to if we were to make that specific shot.  

The Machine learning model is located at: ```ML.py``` in the current workspace. In order to prepare the dataset, I have used the script ```prepare_dataset.py```. The directory ```data``` contains couple of folders. The folder ```images``` includes the original images without any overlay, the ```overlays``` folder contains the images with the overlays and the folder ```masks``` contains the masks of the specific images and overlays. 

Assume currently my datset is almost perfect except for the specific thing I am about to explain. 

Currently, the model detects the line pretty good, but when the lines are really thin and small, the model can't detect them. I have tried many times to use the ```prepare_dataset.py``` and collect overlays so that I collect samples with the correct extended white line when the white-line is thin and small, but yet, nothing worked.  


## Note

It is important to note that you are allowed to look inside of ```data/images``` and view the images in order to understand a little bit better how the game looks like, and you are allowed to go into ```data/overlays``` to see how the overlay should look like. 
</Context>


<Task>
I have created a folder called: ```data/manual_edited_images```
Since I wasn't managed to get a script that could detect small and thin white cue-lines, I have decided to manually create the overlays. I want you to create a script that uses the images inside of ```data/manual_edited_images``` and allows me to paint 2 dots. After painting 2 dots, a drawn extended straight white line will go through this 2 dots (just like the ```prepare_dataset.py``` script creates those overlays). When I have finished to go through all the images inside ```data/manual_edited_images```, the original images will go to ```data/images```, the masks of those images will go to ```data/masks``` and the overlays will go to ```data/overlays```. 


This way, I can manually create a dataset that contains those specific shots I have been missing in my original dataset, and then retrain my model with the new shots. 
</Task>