THIS FILE IS MACHINE GENERATED. IT IS NOT GAURANTEED TO BE CORRECT, ONLY LIKELY TO BE.

# small_personal_projects
## Description
This directory contains a collection of personal projects, including web applications, scripts, and tools for various tasks such as rolling powers from a Celestial Forge document, real-time whisper transcription, training GPT models, working with LangChain, and generating README files. The projects are organized into separate subdirectories, each with its own description and requirements.

# small_personal_projects\celestial_forge_roller_webui
## Description
This directory contains a web application for rolling powers from a Celestial Forge document. It includes a CSV file listing various abilities and items, a list of project dependencies, and a Python script for the web application. The application allows users to roll powers based on selected domains and point costs, with features for undoing rolls, saving and loading roll history, and calculating points based on word count and ratio.

# Easy [Real Time Whisper Transcription](https://github.com/davabase/whisper_real_time)

This repository is meant to make using the Real Time Whisper Transcription repo easier by providing a simple installation method for all of its requirements as well as CUDA accelerated PyTorch. 

It also includes some minor script improvements:
- Implements [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) to reduce transcription delay further
- Minor adjustment to script logic to make the transcription better
- Transcription log saving

### Conda Torch environments are quite large. 
The virtual environment and Whisper model together is over 10GB!

To install the dependencies, install Anaconda and create an isolated Conda environment with 
```
conda env create -p <env path> --file <environment.yml path>
``` 
Then, activate the environment with 
```
conda activate <env path>
```

Now you should be able to execute `python transcribe_demo.py` and have real-time transcription!
>On first run, the script will download the Whisper model to the script's directory.
>Whisper 'medium' is the default; it is ~1.5GB on disk and needs ~4GB of VRAM to run.


# small_personal_projects\gpt_from_scratch
## Description
This directory contains a collection of scripts and files related to training and experimenting with GPT models. It includes scripts for training GPT models on the Shakespeare dataset, as well as a Python script for gradient filtering using the GrokFast library. The directory also contains a Conda environment file for setting up a deep learning environment optimized for PyTorch.

# small_personal_projects\LangChain_scripts
## Description
This directory contains a collection of scripts and Dockerfiles for running LangChain, a Python library for working with large language models, on various platforms. The scripts are designed to work with the Ollama language model and include examples for language model invocation, retrieval-augmented generation, and file analysis. The directory also includes a setup guide for running LangChain-based scripts in a Docker container.

# small_personal_projects\simple_readme_generator
## Requirements
- `pip install requests`
- An OpenAI API-compatible text generation backend at localhost:5000 (I recommend [this one](https://github.com/oobabooga/text-generation-webui) ).
- A Llama 3/3.1 model.
## Description
This directory contains a simple script for generating README.md files for directories and their subdirectories using the OpenAI API. The script requires specific dependencies and can be run by selecting a directory to process.

# .gitignore
## Description
This file specifies files and directories to be ignored by Git in the small_personal_projects directory.

## Ignored Files and Directories
- __pycache__: Directory containing compiled Python bytecode files.
- not_functional: Directory containing non-functional code or test files.