for i in $(seq 1 10); do
    conda deactivate
done

conda activate dm
export PATH="/usr/local/cuda-12/bin:$PATH"
export PYTHONPATH="$(pwd)"