# UniMER
To download UniMER dataset:
under this dir (`datasets`), run:
```bash
cd latex_ocr/datasets

brew install git-xet
git xet install
git clone https://huggingface.co/datasets/wanderkid/UniMER_Dataset

# or
uv tool install hf
hf download wanderkid/UniMER_Dataset --repo-type=dataset --local-dir UniMER_Dataset

# uncompress
cd UniMER_Dataset
unzip UniMER-1M.zip
unzip UniMER-Test.zip
```
Note that, the HWE(handwritten) split is not included in the training set, which is within our expectation.

# Synthetic Dataset
To download the synthetic dataset:
```bash
cd latex_ocr/datasets
hf download PadishahIIIXXX/latex-ocr-dataset --repo-type=dataset --local-dir synth


```
