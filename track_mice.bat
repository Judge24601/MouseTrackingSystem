call C:/Users/greg2/Anaconda3/Scripts/activate.bat  C:/Users/greg2/Anaconda3/envs/dlc-windowsGPU
echo Hello!
SET the_path=%cd%
set /p drive="Enter the path to the folder, not including name: "
set /p name="Enter name of the frame folder/text file: "
cd darknet
C:/Python36/python.exe darknet_video.py -n %name% -d %drive%
C:/Python36/python.exe visualize.py -n %name% -d %drive%
cd ../
rem python head_tail_label.py -n %name%
set /p name="Done!"
