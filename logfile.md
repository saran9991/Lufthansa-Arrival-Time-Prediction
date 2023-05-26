# Logfile - Lufthansa

*Last updated: 12th Mai, 2023*

###### This logfile contains updates to the Lufthansa Project.

## 1. Work on May 9th, 2022 data
The updates made to the dataset are based on the data from May 9th, 2022.

## 2. Assigning Origin Airport: `assign_origin()`
The `assign_origin()` method has been added to calculate the origin airport of an aircraft. This method uses the geopy library to find the nearest airport to the first recorded location of the aircraft in the dataset. However, it is important to note that this method only works if the aircraft is taking off in the specified time frame. If the aircraft is mid-flight in the dataset, this method will return the airport closest to the first recorded location of the aircraft.

To install geopy, run `pip install geopy`.

## 3. Issue: `get_complete_flights()`
The updated `get_complete_flights()` method is not foolproof. Out of the 99 flights returned from a day's data, some had an altitude greater than 7000ft and coordinates located mid-ocean or in another location far from any airport. This issue could be due to the filtering in the `preprocessing.py` file not being based on altitude or geoaltitude. To make the dataset more reliable, it would be valuable to add more parameters to filter complete flights.

## 4. Issue: Assigning Source Airport
If complete flights are not used as training data, assigning a source airport would not be beneficial as the aircraft's location could be close to any arbitrary airport in the vicinity.

## 5. Added `/data` directory 
Added a directory for data under the playground directory for better access to added datasets, such as the airports dataset.

Added an **airports dataset** which contains the coordinates of every airport worldwide.

## 6. Other things to mention related to assigning origin
Another important thing to note is that some aircraft have `onground = True` even when taking off. Earlier, it was assumed that all flights have `onground = False` when taking off but this is not the case for some aircraft. For example, the first recorded location of an aircraft could be an intermediate airport where the aircraft landed and then took off again for its final destination. 

For example, the entire flight could be **( USA - Morroco - Frankfurt )**.  But of course, if we are not focusing on complete flights at all, this would not be an issue. But if the flights are not complete, finding the source airport would be cumbersome and plotting the aircrafts on the globe would not make much sense other than just showing all the flights arriving to Frankfurt.   

Also, in the `get_complete_flights()` method, instead of passing the trajectories, changed the code to pass dataframe, makes things more concise. 

## 7. Created another branch for updating assign_origin
The ``assign_origin()`` method now works faster using R-Tree Index

## 8. Created another branch for plots
Undeveloped branch for plots. Will add more stuff later.

## 9. Created another branch for adding holidays, weekdays and months of the year
This method uses the holidays library to find holidays in Hessen.

To install holidays, run `pip install holidays`.

## 10. The train_models folder contains all the logic to preprocess, load data and train model. All is combined in the train_dl_model.py script

## 11. data_loader.py enables loading the raw-files from the data-folder and processing them in such a way as to allow multi-processing

## 12. trained_models includes the keras-trained deep-learning models, standard-scaler and is used as directory to store eveything which is trained

## 13. Timestamp-feature-engineering: Weekday as one-hot-encoded, holiday as binary, and timestamp as sine- and cosine of seconds within day
and day within year 

## 14. In branch data_loader_edgeflights now most of the edgeflights are also properly preprocessed and integrated into data-loader. 
Only edge-cases between the data-batches aren't integrated yet.

## 15. The preprocessing steps are more memory-efficient now. Data-Frames are main obejects operated on, not Traffic-objects

## 16. Current best model is model_with_cycle_no_montl

## 17. Trained model with dropout layer to reduce overfitting and thinned out training data with using 0.1% of training data
Current best model is model_model_very_thin_dropout

## 18. Plotted L1-Loss by distance in playground notebook

## 19. Updated environment file

## 20. Created another branch for updating preprocessing.py

## 21. Method ``noise_remove`` added
Method removes weird most of the weird altitude and onground changes (yet to be tested on year worth of data)

## 22. Sin and Cosin of track added to aux-columns
Track of the airplane signifies the direction it is heading towards. Sin and cosine fully represent the angle

---
---

