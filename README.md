THIS FILE IS MACHINE GENERATED. IT IS NOT GAURANTEED TO BE CORRECT, ONLY LIKELY TO BE.

# small_personal_projects
## Description
This directory contains a collection of small personal projects, including web applications, scripts, and submodules. The projects are primarily focused on data processing, machine learning, and text generation. They include a web-based application for rolling powers from a Celestial Forge document, a real-time whisper transcription system, a collection of scripts for training and experimenting with GPT models, and a simple README generator project. The directory also includes a submodule for an easy-to-use real-time whisper transcription system.

# small_personal_projects\celestial_forge_roller_webui
## Description
This directory contains a web-based application for rolling powers from a Celestial Forge document. It includes a list of 24 unique abilities and items, categorized into Assistants and Quality Size, and a set of dependencies required for the project to run. The application allows users to input parameters, calculate points, select domains, and roll for powers, with features for undoing rolls and saving roll history.

# Easy [Real Time Whisper Transcription](https://github.com/davabase/whisper_real_time)

This repository makes using the Real Time Whisper Transcription repo easier by providing a simple installation method for all of its requirements as well as CUDA accelerated PyTorch in an isolated Conda environment.  

It also includes some minor script improvements:
- Implements [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) to reduce transcription delay further
- Minor adjustment to script logic to improve the transcription
- Transcription log saving

### Conda Torch environments are quite large. 
The virtual environment and Whisper model together is over 10GB!

### Setup
To install the dependencies, install Anaconda and create an isolated Conda environment with 
```
conda env create -p <env path> --file <environment.yml path>
``` 
Then, activate the environment with 
```
conda activate <env path>
```

Now you should be able to execute `python transcribe_demo.py` or `python transcribe_gui.py`and have real-time transcription!
  >On first transcription run, the script will download the Whisper model to the script's directory.
  >Whisper 'medium' is the default; it is ~1.5GB on disk and needs ~4GB of VRAM to run.

# Building to an executable
Building the GUI script to an executable can make it much more portable and user friendly by removing the need for a Python environment, as well as making it slightly smaller at ~7GB (w/ medium Whisper model).

### Compilation steps:
[Pyinstaller](https://pyinstaller.org/en/stable/) allows easy compilation from a Python script to a packaged, standalone executable for an OS:
1) Install pyinstaller to your Conda environment, via `conda install -c conda-forge pyinstaller` (when the environment is active).
2) Run pyinstaller on the 'transcribe_gui.py' script, ensuring it includes necessary DLLs for CUDA acceleration (which it would otherwise miss), via
```
pyinstaller transcribe_gui.py -D --add-data <venv_path>\Library\bin\cudnn_cnn_infer64_8.dll:. --add-data <venv_path>\Library\bin\cudnn_ops_infer64_8.dll:.
```
> Make sure you fill in your virtual environment's path!  
> Running pyinstaller on a script will produce two folders named `build` and `dist` in your working directory, each containing a folder named after the script.  
> `dist\transcribe_gui` is the directory the functional executable and its dependency folder will be compiled to.  
3) Run your executable, and test out the real time transcription!


# small_personal_projects\gpt_from_scratch
## Description

This directory contains a collection of scripts and files related to training and experimenting with GPT models. It includes scripts for training GPT models on the Shakespeare dataset, implementing gradient filtering techniques, and managing a Conda environment for deep learning and scientific computing tasks. The directory also contains a sample text file with excerpts from Shakespeare's plays.

# small_personal_projects\simple_readme_generator
## Requirements
- `pip install requests`
- An OpenAI API-compatible text generation backend at localhost:5000 (I recommend [this one](https://github.com/oobabooga/text-generation-webui) ).
- A Llama 3/3.1 model.
## Description
This directory contains a simple README generator project, which includes a Python script and documentation for a text generation API. The project allows users to generate a README file for a directory and its contents using a text generation API. It also includes a file containing the API documentation for the Oobabooga API's completions endpoint.

# .gitmodules
## Description
This file defines a submodule named "easy_whisper_real_time" with the following properties:
- Path: easy_whisper_real_time
- URL: https://github.com/christophkogler/easy_whisper_real_time.git