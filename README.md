================================================================================================
This was done on the Mac m3 laptop. 

Install as a single command (concatenated into into single instructions). This scripts installs the "latest" version of python, required libraries for Huggingface 'evaluate' metrics, mog-llm, and airlift-challenge.

Update the conda environment name: dev-airlift

Notes: To specify version of Python;  change 'python' to: 'python==3.13.5'
================================================================================================

In terminal change into the parent directory of your choice.

CONDA_ENV_NAME=dev-airlift && \
echo "1. Creating conda environment: $CONDA_ENV_NAME"
conda create -n $CONDA_ENV_NAME --channel conda-forge -y python setuptools==80.9 pyarrow==20 numpy pandas click matplotlib networkx noise pillow evaluate transformers seaborn scipy scikit-learn statsmodels ollama pytest && \
conda activate $CONDA_ENV_NAME && \
pip install bert-score rouge-score codebleu absl-py nltk flake8 gym pyglet redis pytest_redis pettingzoo opencv_python msgpack_numpy msgpack graphviz ordered-set odfpy && \
pip install --no-deps tree-sitter==0.23.1 tree-sitter-python && \
pip install --upgrade evaluate datasets && \
git clone https://github.com/airlift-challenge/airlift && cd airlift && \
touch requirements_dev.txt && echo "" > requirements_dev.txt && sed -i '' 's/self._np_random.randint(/self._np_random.integers(/g' ./airlift/envs/generators/cargo_generators.py && sed -i '' '355 s/top_right + bottom_left/bottom_left + top_right/g' ./airlift/envs/renderer.py && \
pip install --use-pep517 -e .


To remove the above:
Change to parent directory and set conda environment name.
CONDA_ENV_NAME=dev-airlift
rm -rf ./airlift && \
conda deactivate && conda env remove --name $CONDA_ENV_NAME -y


================================================================================================
Ollama install:
  Download and install: https://ollama.com/download/Ollama.dmg
  After ollama install, execute the following commands in the terminal:
ollama pull phi3:3.8b && \
ollama pull gemma3:4b && \
ollama pull codellama:7b && \
ollama pull mistral:7b && \
ollama pull qwen3:8b && \
ollama pull deepseek-r1:8b && \
ollama pull mistral-nemo:12b && \
ollama pull gemma3n:e4b && \
ollama pull gemma3:12b-it-qat && \
ollama pull phi4:14b && \
ollama pull codestral:22b && \
ollama pull zephyr:7b-beta-fp16 && \
ollama pull gemma3:27b && \
ollama pull gemma3:27b-it-qat && \
ollama pull llama3.3:70b
  
  For installation details, see slide 5 of the Install-and-user-guide.odp
