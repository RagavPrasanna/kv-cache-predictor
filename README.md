# ML-Based KV Cache Prediction for LLM Serving

An Operating Systems project exploring whether machine-learning predictions of request KV-cache usage can improve memory-aware scheduling for large language model (LLM) serving. The project trains LightGBM and quantile-regression models on request traces, then compares a predictive scheduler with a max-reservation baseline in simulation.

The report describes experiments on 500 ShareGPT requests with OPT-1.3B and reports up to 566.7% more admitted requests than the baseline in the simulated 200-block pool. See [`Report.pdf`](Report.pdf) for methodology, results, and limitations.

## Repository contents

- [`Scripts/Script1.ipynb`](Scripts/Script1.ipynb) — ShareGPT preprocessing and vLLM instrumentation.
- [`Scripts/Script2.ipynb`](Scripts/Script2.ipynb) — exploratory analysis, model training, and prediction/scheduling plots.
- [`Scripts/Script3.ipynb`](Scripts/Script3.ipynb) — predictive scheduler simulation and additional evaluations.
- [`graphs_results/`](graphs_results/) — example CSV input, trained model artifact, and result plots.
- [`Report.pdf`](Report.pdf) — final project report.

## Running the notebooks

The notebooks were designed for Google Colab. Script 1 requires a Linux GPU runtime (the project used a Colab T4); Scripts 2 and 3 can run on CPU.

1. Open the notebooks in `Scripts/` in order. For data collection, use a compatible Colab GPU runtime and install the dependencies listed in the first cell of Script 1.
2. Script 2 reads `kv_cache_data.csv` from the current working directory. To reproduce the checked-in training/evaluation run, upload or copy [`graphs_results/kv_cache_data.csv`](graphs_results/kv_cache_data.csv) into the notebook's working directory.
3. Script 2 writes `models.pkl` and plots to the working directory. Script 3 reads both `kv_cache_data.csv` and `models.pkl`; for the checked-in example, upload/copy [`graphs_results/models.pkl`](graphs_results/models.pkl) and the CSV before running it.
4. The generated plots can be compared with the examples in `graphs_results/`.

Install the shared Python dependencies for Scripts 2 and 3 with:

```bash
python -m pip install -r requirements.txt
```

Script 1 also depends on vLLM and GPU/CUDA support. The notebook currently requests `vllm==0.3.3`; vLLM is Linux/CUDA-specific and its internal scheduler APIs can vary between releases. Run it in a compatible environment rather than a standard Windows Python environment.

> **Reproducibility note:** The checked-in Script 1 notebook contains dataset preprocessing and vLLM instrumentation setup, but does not include a complete trace-export step. Use the checked-in CSV for Scripts 2 and 3, or complete/verify the data-collection and export flow before attempting to regenerate it.

## Results and limitations

The report's scheduling and safety results are based on a simulation and a 500-request trace; they should not be interpreted as production vLLM performance guarantees. The project discusses dataset size and real scheduler integration among its limitations.

`graphs_results/models.pkl` is a Python pickle. Only load it if you trust the source; pickle files can execute code when deserialized.

## GitHub

Source repository: <https://github.com/RagavPrasanna/kv-cache-predictor>
