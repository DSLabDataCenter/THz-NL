<h1 align="center">
Nonlinear Behaviors of Transceivers for Terahertz Communications: Data Sets and Models
</h1>

This repository provides the datasets and implementation code accompanying the paper *“Nonlinear Behaviors of Transceivers for Terahertz Communications: Data Sets and Models.”*  

## Data Format

The **Terahertz Nonlinear Dataset** is located under:  
```
THz-NL/dataset/DATA_W_D_H_band/Baseband_input_output/
```
This directory contains two main folders:
```
Baseband_input_output/
  ├── input/
  │ ├── D_band/
  │ ├── W_band/
  │ └── H_band/
  └── output/
    ├── D_band/
    ├── W_band/
    └── H_band/
```

- **Input files** are stored under:
```
input/{Band}/PA_input_BS_60G_{CarrierFrequency}G_{QAM}QAM_{FrequencyBand}.mat
```
- **Output files** are stored under:
```
output/{Band}/PA_baseband_{CarrierFrequency}G_{QAM}QAM_{FrequencyBand}_{n}.mat
```
where:  
- `{Band}` ∈ {`D_band`, `W_band`, `H_band`}  
- `{CarrierFrequency}` denotes the carrier frequency (e.g., 1, 2, 4).  
- `{QAM}` represents the modulation order (e.g., 16, 64).  
- `{FrequencyBand}` identifies the operating band (e.g., `D`, `W`, `H`).  
- `{n}` ∈ {1, 2, 3, 4, 5} is the **output index** — each input file corresponds to **five** distinct output files, representing repeated measurements under different amplifier operating conditions.
---

Each `.mat` file contains a **complex-valued time-domain signal**, represented as:

$$x(t) = x_{\text{Re}}(t) + j\,x_{\text{Im}}(t), \quad t = 1, 2, \ldots, T $$

where:  
- $ x_{\text{Re}}(t) $ and $ x_{\text{Im}}(t) $ are the real and imaginary parts, respectively;  
- $ T $ denotes the number of time samples.


