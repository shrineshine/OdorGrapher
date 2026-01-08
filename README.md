# OdorGrapher

## Set up the Python environment

```
conda create -n odorgrapher python=3.9
conda activate odorgrapher
pip install -r requirements.txt
```

## Prepare dataset
The datasets used in this study have been deposited in a public repository and will be made publicly available upon publication of the manuscript.
Please download the following files:
- `goodscents_merged.csv`
- `leffingwell_merged.csv` 
- `flavordb_merged.csv`

After downloading, place the three files into the `data/raw/`. 
The directory structure should look like:
```
OdorGrapher/
├── codes/
├── data/
│   └── raw/
│       ├── goodscents_merged.csv
│       ├── leffingwell_merged.csv
│       └── flavordb_merged.csv
```

```
python codes/prepare_data.py
```

## Model training

```
python codes/main.py
```
