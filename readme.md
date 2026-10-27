<h1 align="center" style="font-weight: bold; font-size: 2.5rem;"> Adversarial Robustness of Retrieval-Based Fact-Checking: <span style="font-size: 1.5rem; font-weight: normal; color: #555;">Detecting and Mitigating Evidence Contamination</span> </h1> <div align="center">
Alamgir Munir Qazi<sup>1</sup>   John P. McCrae<sup>2</sup>   Jamal Abdul Nasir<sup>1</sup>

<sup>1</sup>School of Computer Science, University of Galway, Ireland <sup>2</sup>Research Ireland Insight Centre and ADAPT Centre, University of Galway, Ireland

</div> <div class="content has-text-justified"> <p> Retrieval-based fact-checking systems ground verification in large document collections, creating a critical vulnerability: adversarial evidence contamination. We demonstrate that factually inconsistent but contextually coherent evidence generated to satisfy both retrieval-similarity and linguistic-fluency constraints can systematically mislead verification models. Contaminating just 3.1-7.4% of training corpora with five label-specific strategies causes adversarial content to penetrate up to 35.2% of top-k retrievals, degrading verification accuracy by up to 46.9% relative F1. Existing retrieval-stage defenses, validated against gradient-optimized or syntactically anomalous attacks, transfer poorly to this threat model: detection rates collapse and F1 recovery exhibits high variance across seeds. We propose GLAD (Geometric-Lexical Adversarial Defense), which fuses dense embedding peer-coherence with lexical distributional isolation. Across two benchmarks, two embedding models, and five random seeds, GLAD achieves the highest detection rates (70-86%) and the most consistent F1 recovery, reclaiming 73-80% of contamination-induced degradation on LIAR-RAW while approaching clean baselines on RAWFC. Crucially, unlike defenses that trade clean performance for robustness, GLAD imposes negligible cost on uncontaminated corpora. </p> </div>


## Code

### Prerequisites

#### Dataset

Create a folder named `dataset` and place it in the root directory.

Download the source datasets from https://github.com/Nicozwy/CofCED and place them in that directory:

```
dataset/
├── RAWFC/{train,val,test}/*.json
└── LIAR-RAW/{train,val,test}.json
```

The generated adversarial evidence files are read from the same folders and are
configured in `ADVERSARIAL_PATHS` at the top of `defence/cache-main.py`:

```
dataset/RAWFC/rawfc-{train,val,test}-gemma3-adversarial.json
dataset/LIAR-RAW/liarraw-{train,val,test}-gemma3-adversarial.json
```

Only the **train** files are used by the reported experiments — contamination is
injected into the training index, while the validation and test retrievals stay clean
and unfiltered.

### Setup

#### Using UV (Preferred)

```
uv venv venv --python 3.10

source venv/bin/activate

uv pip install "torch>=2.6.0" torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

uv pip install faiss-gpu-cu12

uv pip install -r requirements.txt
```

#### Using Conda

```
conda create --prefix venv python=3.10

conda activate venv/

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install faiss-gpu -c conda-forge

pip install -r requirements.txt
```

> **Caveat:** `transformers` must be pinned to `4.45.0`. The latest release (`4.56.1`)
> breaks `Alibaba-NLP/gte-Qwen2-1.5B-instruct`.


## How to run

```
python defence/cache-main.py --dataset RAWFC --model nomic --seed 43
```

## License

Code is released for research use.
