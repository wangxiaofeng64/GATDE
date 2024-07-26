
# GATDE: A graph attention network with diffusion-enhanced protein-protein interaction for cancer classification

## Project Description

This project aims to develop a Diffusion-Enhanced Graph Attention Network (DEGAT) for classifying cancer types based on multi-omics data. The primary goal is to integrate clinical data, protein-protein interaction (PPI) networks, and reverse-phase protein array (RPPA) data to accurately predict cancer subtypes.

## Project Structure

The project is organized as follows:

```
.
├── data
│   ├── brain
│   │   ├── clinical_data_of_brain.csv
│   │   ├── ppi_network_of_brain.csv
│   │   └── RPPA_data_of_brain.csv
│   ├── BRCA
│   │   ├── clinical_data_of_BRCA.csv
│   │   ├── ppi_network_of_BRCA.csv
│   │   └── RPPA_data_of_BRCA.csv
│   ├── Lung
│   │   ├── clinical_data_of_Lung.csv
│   │   ├── ppi_network_of_Lung.csv
│   │   └── RPPA_data_of_Lung.csv
│   ├── pan-cancer
│   │   ├── clinical_data_of_pan.csv
│   │   ├── ppi_network_of_pan_cancer.csv
│   │   └── RPPA_data_of_pan_cancer.csv
│   └── point
├── venv
├── gat.py
├── models.py
├── README.md
└── requirements.txt
```

## Data Description

Each cancer type directory (`brain`, `BRCA`, `Lung`, `pan-cancer`) contains three CSV files:

- `clinical_data_of_<cancer_type>.csv`: Clinical data of patients.
- `ppi_network_of_<cancer_type>.csv`: Protein-Protein Interaction (PPI) network data.
- `RPPA_data_of_<cancer_type>.csv`: Reverse-Phase Protein Array (RPPA) data.

## Scripts

- `gat.py`: Implements the Graph Attention Network (GAT) layer used in the model.
- `models.py`: Contains the main DEGAT model implementation, including data preprocessing, model building, training, and evaluation.

## Installation

1. Clone the repository.
2. Create and activate a virtual environment:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

## Usage

1. Prepare your data by placing it in the appropriate directories under `data/`.
2. Run the `models.py` script to train and evaluate the model.

```sh
python models.py
```

## Requirements

- Python 3.7+
- TensorFlow
- Scikit-learn
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn

Refer to `requirements.txt` for the full list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This research was conducted at Nankai University, supported by the college of software and School of Mathematical Sciences.
