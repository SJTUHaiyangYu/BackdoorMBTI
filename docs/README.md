# How to build the docs using ReadTheDocs

## Step 0: Install the requirements
```shell
➜ conda create -n docs python=3.10
➜ conda activate docs
➜ pip install -r requirements.txt
```

## Step 1: Build the docs locally

```shell
make html
make clean
```