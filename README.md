<div align=center>
  <h1>2D-Neural-Sensorimotor-Control</h1>

![Python](https://img.shields.io/badge/Python-3776AB?logo=Python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=NumPy&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=Numba&logoColor=white)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://opensource.org/licenses/MIT)

</div>

A package for applying neural sensorimotor control law and consensus sensing algorithm for computational octopus arm model simulated in PyElastica.

## Dependency & installation

### Requirements
  - Python version: 3.9
  - Additional package dependencies include: [NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html), [SciPy](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide), [Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html), [Matplotlib](https://matplotlib.org/stable/users/explain/quick_start.html), [tqdm](https://tqdm.github.io/), [PyElastica](https://github.com/GazzolaLab/PyElastica), [H5py](https://docs.h5py.org/en/stable/), and [Click](https://click.palletsprojects.com/en/stable/) (detailed in `pyproject.toml`)

### Installation

Before installation, create a Python virtual environment to manage dependencies and ensure a clean installation of the **2D-Neural-Sensorimotor-Control** package.

1. Create and activate a virtual environment: (If you have already created a virtual environment for **2D-Neural-Sensorimotor-Control**, directly activate it.)

    ```properties
    # Change directory to your working folder
    cd path_to_your_working_folder

    # Create a virtual environment of name `myenv`
    # with Python version 3.9
    conda create --name myenv python=3.9

    # Activate the virtual environment
    conda activate myenv

    # Note: Exit the virtual environment
    conda deactivate
    ```

2. Install Package: (two methods)

    ```properties
    ## Need ffmpeg installed from Conda
    conda install conda-forge::ffmpeg
    
    ## Install directly from GitHub
    pip install git+https://github.com/tixianw/2D-Neural-Sensorimotor-Control.git

    ## Or clone and install
    git clone https://github.com/tixianw/2D-Neural-Sensorimotor-Control.git (download directly if cannot clone)
    cd 2D-Neural-Sensorimotor-Control
    pip install .

<details>

<summary> Click me to expand/collapse developer environment setup </summary>

## Developer environment setup

1. Clone and install development dependencies:
    ```properties
    git clone https://github.com/tixianw/2D-Neural-Sensorimotor-Control.git
    cd 2D-Neural-Sensorimotor-Control
    pip install pip-tools
    ```

2. Generate development requirements file:
    ```properties
    pip-compile pyproject.toml --output-file=requirements.txt
    ```

</details>

## Example

Please refer to [`examples`](https://github.com/tixianw/2D-Neural-Sensorimotor-Control/tree/main/examples) directory and learn how to use this **2D-Neural-Sensorimotor-Control** package. Three examples are provided. First nevigate to one of the example folders before running the scripts.
  - [`neuromuscular_control`](https://github.com/tixianw/2D-Neural-Sensorimotor-Control/tree/main/examples/neuromuscular_control) demonstrates three cases: `LM_reach`, `LM_point` and `TM_reach` which illustrate how longitudinal muscles and transverse muscle actuate the neuromuscular arm model to reach or point towards a static target.
    First nevigate to the `neuromuscular_control` folder:
    ```
    cd examples/neuromuscular_control/
    ```
    Run the following commands to simulate one of the three cases: `LM_reach`, `LM_point` or `TM_reach`. For instance:
    ```
    python run_simulation.py --case LM_reach
    ```
    When simulation script finishes, run the following commands to plot the results:
    ```
    python plotting_neuromuscular.py --case LM_reach
    ```

  - [`sensing`](https://github.com/tixianw/2D-Neural-Sensorimotor-Control/tree/main/examples/sensing) showcases how a straight or bent static arm performs chemosensing and proprioception locally from the suckers and collectively estimate where the target locates(food source that diffuses chemical concentration).
    First nevigate to the `sensing` folder:
    ```
    cd examples/sensing/
    ```
    Run the following commands to simulate one of the two cases: `straight` or `bend`. For instance:
    ```
    python run_simulation.py --case straight
    ```
    When simulation script finishes, run the following commands to plot the results:
    ```
    python plotting_sensory.py --case straight
    ```

  - [`sensorimotor_control`](https://github.com/tixianw/2D-Neural-Sensorimotor-Control/tree/main/examples/sensorimotor_control) provides a simulation results for the end-to-end sensorimotor control framework where the arm simultaneously performs sensing and neuromotor feedback control to reach towards the estimated target.
    First nevigate to the `sensorimotor_control` folder:
    ```
    cd examples/sensorimotor_control/
    ```
    Run the following commands to simulate the sensorimotor control:
    ```
    python run_simulation.py
    ```
    When simulation script finishes, run the following commands to plot the results:
    ```
    python plotting_sensorimotor.py
    ```

## License

This project is released under the [MIT License](https://github.com/tixianw/2D-Neural-Sensorimotor-Control/blob/main/LICENSE).

## Citation

@article{wang2024neural,
  title={Neural models and algorithms for sensorimotor control of an octopus arm},
  author={Wang, Tixian and Halder, Udit and Gribkova, Ekaterina and Gillette, Rhanor and Gazzola, Mattia and Mehta, Prashant G},
  journal={arXiv preprint arXiv:2402.01074},
  year={2024}
}


<!-- ## Contributing

1. Fork this repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m "feat: Add some amazing feature"`)
5. Push to the feature branch (`git push origin feat/amazing-feature`)
6. Open a Pull Request -->
