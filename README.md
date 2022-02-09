# Mountain Car

Simulating an episode is done in Python so not much speedup from CUDA. The main bottleneck is the training where we update all the weights based on the episode actions and results. Training performance in CUDA is about 1000x faster than Python for 10 workers.


## Running

1. Install python3, cmake, OpenAI Gym, and CUDA
2. Make build dir and build using cmake
3. Add the output directory to the PYTHONPATH. For example, add `./build/Debug/` to the `PYTHONPATH`
4. Run the python script `python3 src/main.py`
   1. Maybe need to `cd ./build/Debug` first and run `python3 ..\..\src\main.py` instead 
