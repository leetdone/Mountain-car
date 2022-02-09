# Mountain Car

Simulating an episode is done in Python so not much speedup from CUDA. The main bottleneck is the training where we update all the weights based on the episode actions and results. Training performance in CUDA is about 1000x faster than Python for 10 workers.
![cmake](https://user-images.githubusercontent.com/94505781/153123897-be73e768-7ac5-4d6a-96ca-f1ccd19fe4b2.PNG)
<p align="center">
   <img src="https://user-images.githubusercontent.com/94505781/153123897-be73e768-7ac5-4d6a-96ca-f1ccd19fe4b2.PNG" width="48">
</p>

![cudaMultithreadDesign](https://user-images.githubusercontent.com/94505781/153123906-445bb055-32a5-4389-8c60-0ffbdee3cbd7.PNG)

![environment](https://user-images.githubusercontent.com/94505781/153123941-71a2faff-9b42-44cd-8a7a-0b49a289b682.PNG)
![mountainCar](https://user-images.githubusercontent.com/94505781/153123950-4b9db3e0-b58b-4dc2-a953-36e3578c9df6.PNG)
![pythonWorkFlow](https://user-images.githubusercontent.com/94505781/153123975-86ec864e-a063-43e7-a529-6bce47efdf74.PNG)

## Running

1. Install python3, cmake, OpenAI Gym, and CUDA
2. Make build dir and build using cmake
3. Add the output directory to the PYTHONPATH. For example, add `./build/Debug/` to the `PYTHONPATH`
4. Run the python script `python3 src/main.py`
   1. Maybe need to `cd ./build/Debug` first and run `python3 ..\..\src\main.py` instead 
