# SupCon-LFA

1. The file "main_supcon.py" is for the model's pretraining; the file "main_linear.py" is for training the model's classifier and testing the entire model.
2. The "save/SupCon/path_models" folder contains models for the "CBIS-DDSM" and "INbreast" datasets, which can be directly run using
	```python
	python main_linear.py --dataset path --batch_size 4 --epochs 60 --ckpt save/SupCon/path_models/INbreast/last.pth --mean 0.17666 --std 0.2539 --data_folder_train ../INbreast/train/ --data_folder_test ../INbreast/test/
	```
	or
	```python
	python main_linear.py --dataset path --batch_size 32 --epochs 60 --ckpt save/SupCon/path_models/CBIS-DDSM/last.pth --mean 0.51155 --std 0.1848 --data_folder_train ../CBIS_raw_data/CBIS_train/ --data_folder_test ../CBIS_raw_data/CBIS_test/
	```

3. You can also first pre-train the model using "main_supcon.py" and then train the model's classifier and test the entire model using "main_linear.py".
	For example with the CBIS-DDSM dataset:
	```python
	python main_supcon.py --dataset path --mean 0.51155 --std 0.1848 --data_folder ../CBIS_raw_data/CBIS_train/ --batch_size 16 --epochs 60 --size 448 --trial 0 --temp 0.2
	```
	then 
	```python
	python main_linear.py --dataset path --batch_size 32 --epochs 60 --ckpt save/SupCon/path_models/SupCon_path_resnet50_lr_0.005_decay_0.0001_bsz_16_temp_0.2_trial_0/last.pth --mean 0.51155 --std 0.1848 --data_folder_train ../CBIS_raw_data/CBIS_train/ --data_folder_test ../CBIS_raw_data/CBIS_test/
	```
4. The INbreast dataset is randomly split into training and testing sets at a 4:1 ratio, and to ensure fairness in comparison, all experiments use the same training and testing sets.
