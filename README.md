Datasets and Code of "Nonlinear Behaviors of Transceivers for Terahertz Communications: Data Sets and Models"
---

## Experimental Platform

Fig. 1 illustrates the experimental setup of the PA test bench, composed of a PC, an arbitrary waveform generator (AWG), mixers, a power amplifier (PA), transmitter and receiver horn antennas, and an oscilloscope.
The baseband input signals are 16/64-quadrature amplitude modulation (16-QAM/64-QAM) orthogonal frequency division multiplexing (OFDM) signals. 
At the transmitter side, the baseband signal is loaded by the AWG, and then up-coverted to different test frequency bands through a mixer. 
The corresponding radio frequency signal is amplified by the PA and subsequently radiated through the transmitter horn antenna. 
Following a 1.0-meter wireless transmission, the signal is captured by the receiver horn antenna. 
The received signal is then down-converted to baseband using a mixer synchronized with the same local oscillator and frequency multiplier, and finally captured by an oscilloscope. 
The description of the test signals in the experiments are summarized in Table 1.

<div align="center">
  
### Table 1 Descriptions of the test signals in the experimental platform

| | Frequency Band | Carrier Frequency | Bandwidth |
|----------------|----------|------------|------------|
| PA W 1G | W | 96 GHz | 1 GHz |
| PA W 2G | W | 96 GHz | 2 GHz |
| PA W 4G | W | 96 GHz | 4 GHz |
| PA D 1G | D | 141 GHz | 1 GHz |
| PA D 2G | D | 141 GHz | 2 GHz |
| PA D 4G | D | 141 GHz | 4 GHz |
| PA G 1G | G | 228 GHz | 1 GHz |
| PA G 2G | G | 228 GHz | 2 GHz |
| PA G 4G | G | 228 GHz | 4 GHz |

</div>

## Nonlinearity Analysis
Without loss of generality, we take 64-QAM OFDM input signals as an example for illustrating the PA’s nonlinear behaviors. 
The input signal bandwidth is configured to 1 GHz, 2 GHz and 4 GHz, respectively.
The carrier frequency is set to be 96 GHz, 141 GHz and 228 GHz, corresponding to the W, D and G frequency band, respectively. 
Fig. 2 shows the measured amplitude/amplitude (AM/AM) characteristics of the PAs under different frequency bands and bandwidths. 
The observed spread in the AM/AM response characteristics demonstrates the presence of memory effects in the PAs. 
Furthermore, the AM/AM response reveals a positive correlation between dispersion magnitude and input signal bandwidth. 
This phenomenon indicates enhanced memory effects in power amplifiers when operating under ultra-wideband conditions, with the effect severity scaling proportionally to the bandwidth.


To further comprehensively evaluate in-band signal distortions and out-of-band spectral leakage induced
by PA nonlinear effects, we further analyze constellation diagrams and power spectral density
(PSD) figures of PA outputs, as illustrated in Fig. 3 and Fig. 4, respectively. The constellation diagrams
quantify phase and amplitude deviations within the operational bandwidth, while the PSD figures
explicitly reveal harmonic distortions and intermodulation products beyond the allocated spectrum.
From Fig. 3, it is observed that the 64-QAM constellation exhibits pronounced cluster dispersion,
characterized by radial spreading from AM/AM distortions and phase rotations from amplitude/phase
(AM/PM) nonlinearity. Such distortions directly deteriorate the EVM performance, destabilizing highorder
64-QAM demodulation thresholds. Furthermore, as shown in Fig. 4, pronounced out-of-band
spectral leakage is observed. Taking 1 GHz input signal for instance, third-order harmonic emissions
exceeding -26 dBc at 1 GHz offsets, directly violating 3GPP spectral mask requirements and inducing
adjacent-channel interference. Concurrently, severe in-band imbalance manifests as asymmetric constellation
distortion. The trend agrees with the experimental results shown in Fig. 3.


## Proposed Method. 

We propose an **Augmented Real-Valued Multi-scale Convolutional Transformer Network (ARVMCTN)** to model the nonlinear behaviors and memory effects of terahertz (THz) transceivers across different frequency bands and bandwidths. Specifically, we first convert the complex baseband input signals into an augmented real-valued representation by concatenating their real and imaginary parts, amplitudes, and higher-order nonlinear terms, enabling the model to comprehensively capture signal characteristics. We then design a two-stage feature extraction structure: the first stage employs a multi-scale convolutional module composed of three parallel 2D convolutional branches with different kernel sizes to extract local features under multiple receptive fields, while the second stage utilizes stacked multi-head self-attention Transformer encoder layers to capture long-range dependencies and global temporal dynamics. To enhance the model’s generalization ability across diverse domains, we introduce a lightweight **parameter modulation mechanism**, where domain parameters such as frequency band, carrier frequency, and bandwidth are one-hot encoded and passed through two small multilayer perceptrons to generate per-channel scaling (γ) and shifting (β) factors. These are applied to intermediate features to achieve domain-adaptive modulation. Finally, the contextual features output by the Transformer encoder are fed into a fully connected regression layer to predict the real and imaginary parts of the output signal. Through this design, our model effectively captures both local nonlinearities and long-range dependencies, demonstrating superior modeling accuracy and robustness across multi-band and ultra-wideband THz systems.



