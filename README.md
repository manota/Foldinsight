First, install Anaconda from https://www.anaconda.com/download

Then, create a new conda environment with Python 3.10:
```
conda create -n myenv python=3.10
```

Install the required packages:
```
conda install -c anaconda numpy -y
conda install -c anaconda pandas -y
conda install -c anaconda seaborn -y
conda install -c anaconda scikit-learn -y
conda install -c anaconda joblib -y
conda install -c anaconda scipy -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge biopython -y
conda install -c conda-forge pyarrow -y
conda install -c numba numba -y
conda install -c conda-forge pdb2pqr -y
conda install -c salilab modeller -y
conda install -c conda-forge openmm -y
conda install -c conda-forge gcc=12.1.0 -y
conda install cmake -y
conda install -c conda-forge pdbfixer -y
```
Caution: "modeller" needs a license, you can get a free one from https://salilab.org/modeller/ <br>
and install "chimera" following the instructions at https://www.cgl.ucsf.edu/chimera/ <br>

Clone the repository:
```
gh repo clone manota/Foldinsight
```

Then, navigate to the Foldinsight directory:<br>
0. (optional) Use `0_main_optimize_structure.py` to optimize protein structures.

    | Argument         | Description                                                                                           | Default                               |
    | ---------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------- |
    | `--input-dir`    | Directory containing original PDB files to optimize                                                   | `./00ori_structure`                   |
    | `--output-dir`   | Directory where optimized PDB files (`*_opt.pdb`) will be placed                                      | `./01structure`                       |
    | `--chimera-path` | Full path to the Chimera executable (auto-detected under `$HOME/*/UCSF*/bin/chimera` if not provided) | First match under your home directory |
    
    example:
    ```
    python 0_main_optimize_structure.py --input-dir ./your_pdbs_dir \
      --output-dir ./optimized_pdbs_dir \
      --chimera-path /home/ota/.local/UCSF-Chimera64-1.19/bin/chimera
    ```

1. Use `1_main_mf.py` to get fixed-length vector expressing protein structure (Molecular Fields) from your proteins.

    | Argument           | Description                                                                                     | Default             |
    | ------------------ | ----------------------------------------------------------------------------------------------- | ------------------- |
    | `--input-dir`      | Directory containing your optimized PDB files (will look for `*.pdb`)                           | `./01structure`     |
    | `--output-dir`     | Directory where Molecular Field outputs (`*.csv`, `*.feather`) will be written                  | `./MolecularFields` |
    | `--step-size`      | Lattice step size in Å for the energy grid box                                                  | `1.0`               |
    | `--leave-temp-dir` | If set, do **not** delete the temporary work directory and print its path at the end of the run | *disabled*          |

   example:
    ```
    python 1_main_mf.py --input-dir ./your_optimized_pdbs_dir \
      --output-dir ./your_molecular_fields_dir \
      --step-size 1.0
    ```
3. Use `2_main_cv.py` to construct and evaluate prediction model using Molecular Fields.
    
    | Argument         | Description                                                                                | Default                          |
    | ---------------- | ------------------------------------------------------------------------------------------ | -------------------------------- |
    | `--input-X-dir`  | Directory containing your Molecular Field feather files (`vdW.feather`, `coulomb.feather`) | `./MolecularFields`              |
    | `--input-Y-path` | Path to the CSV file with your observed Y-values (functionality)                           | `./MolecularFields/y_values.csv` |
    | `--output-dir`   | Directory where the plot (`observedVSpredicted.svg`) will be saved                         | `./results`                      |

   example:
    ```
    python 2_main_cv.py --input-X-dir ./your_molecular_fields_dir \
      --input-Y-path ./your_y_values.csv \
      --output-dir ./your_results_dir
    ```
5. Use `3_main_visualize.py` to visualize important region of protein as functionality.
    
    | Argument                    | Description                                                                                               | Default                               |
    | --------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------- |
    | `--input-X-dir`             | Directory containing your Molecular Field mapping files (`vdW.feather`, `coulomb.feather`, `gridbox.csv`) | `./MolecularFields`                   |
    | `--input-Y-path`            | Path to the CSV file of observed functionality values (`y_values.csv`)                                    | `./MolecularFields/y_values.csv`      |
    | `--representative-pdb-path` | Path to a single PDB file that will serve as the reference structure for Chimera visualization            | `./01structure/p_0000.pdb`            |
    | `--output-dir`              | Directory where the Chimera overlay script and final images will be written                               | `./results`                           |
    | `--chimera-path`            | Full path to the Chimera executable (auto-detected under `$HOME/*/UCSF*/bin/chimera` if not provided)     | First match under your home directory |
    | `--leave-temp-dir`          | If set, do **not** delete the temporary working directory and print its path at the end                   | (disabled)                            |

   example:
    ```
    python 3_main_visualize_important_regions.py \
      --input-X-dir ./MolecularFields \
      --input-Y-path ./MolecularFields/y_values.csv \
      --representative-pdb-path ./01structure/p_0000.pdb \
      --output-dir ./results \
      --chimera-path $HOME/.local/UCSF-Chimera64-1.19/bin/chimera \
    ```

