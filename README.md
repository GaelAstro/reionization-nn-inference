# Neural Network Inference of Reionization Parameters

M1 SUTS Master's LIU - Observatoire de Paris (LUX Laboratory)

**Supervisor:** Dr. Benoit Semelin

## What This Does

Uses neural networks to predict astrophysical parameters (star formation timescale, minimum halo mass, UV escape fraction) from 21cm power spectra during the Epoch of Reionization.

**Data:** LoReLi II database (9,828 LICORICE simulations)

## Key Results

- 3 parameters together: ~12-22% error
- 1 parameter alone: **0.87% error** 
- Found strong degeneracy between $\tau_{SF}$ and log $M_{min}$ (correlation -0.82)

## Installation
```bash
pip install numpy tensorflow matplotlib scikit-learn
```

## Files

- `train_nn.py` - Main training script with everything
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Data

LoReLi II data not included. Contact Dr. Benoit Semelin for access.

## Citation
```bibtex
@mastersthesis{ferreirarodriguez2025,
  author = {Ferreira Rodriguez, Gaël},
  title = {Neural Network Inference of Reionization Parameters from 21cm Power Spectra},
  school = {Observatoire de Paris},
  year = {2025},
  type = {M1 Laboratory Insertion Unit Report},
  note = {Master Sciences de l'Univers et Technologies Spatiales (SUTS)}
}
```

## Contact

Gaël Ferreira Rodriguez - M1 SUTS, Observatoire de Paris
gael.ferreira-rodriguez@obspm.fr
