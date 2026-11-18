# Hybrid Lite: a lightweight CNN-XGBoost intrusion detector for CAN-bus attacks

Hybrid Lite classifies a CAN-bus frame as normal or attack in about 0.34 ms on a CPU, inside the 0.5 ms real-time budget of the bus. It combines a small CNN for feature extraction with XGBoost for the decision. The CNN runs as an int8 TensorFlow Lite model and the XGBoost trees run as compiled C code. This repository has the model code and a SUMO city simulation in which the model detects and mitigates five simultaneous CAN-bus attacks.

## The model

`DCNN_Lite -> 64-d embedding -> int8 TFLite -> compiled XGBoost (100 trees, depth 6)`

- Input: a 29 x 29 frame made from the CAN-ID bits of 29 consecutive messages.
- DCNN_Lite is a small CNN with 34,962 parameters, including its training head. After training, the softmax head is removed and the 64-d embedding is kept.
- XGBoost is trained on the embeddings, then exported as C source and compiled with gcc.
- Attacks covered: DoS, Fuzzy, Gear spoofing, RPM spoofing and Replay.

## Results

| Item | Result |
|---|---|
| Mean F1 over the five datasets | 0.9936 |
| Mean false negative rate | 0.83% |
| Median latency | 0.3386 ms per frame (0.68 of the 0.5 ms budget) |
| Replay F1 | 0.9701 |
| Unified model in the SUMO city (all five attacks) | 94.7% of attack hits detected |
| Crashes of attacked vehicles in the SUMO city | 56% fewer per 1,000 hits |

Latency was measured on a shared 2-core Colab CPU and depends on the machine. The model detects the attack types it was trained on. A model trained on one attack largely misses the others, so train on all attack types. Replay is detected poorly by the unified model, because the input has identifier bits only and no timing.

## Repository contents

| Path | Purpose |
|---|---|
| `hybrid_lite.py` | Trains and evaluates the model on the five datasets and measures latency |
| `requirements.txt` | Python packages for `hybrid_lite.py` |
| `SUMO_simulation_of_city_traffic_and_attack_mitigation.zip` | The simulated city, the five attacks, the combined attack and the defence. Unzip it and read its `README.md` |

## Run the model script

1. Install Python 3.10 or newer and a C compiler (gcc or clang) on the PATH.
2. Install the packages: `pip install -r requirements.txt`
3. Download the Car-Hacking dataset of Song et al. (HCRL, Korea University) and put `DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv` and `RPM_dataset.csv` in `./data`. The data is not included here. Check its terms of use.
4. Run: `python hybrid_lite.py --data-dir ./data --out-dir ./results`

The script makes the Replay dataset itself from the normal traffic of the DoS file (300 bursts of 204 messages at 1 ms) and saves it as `data/Replay_dataset.csv`.

Outputs in `./results`:
- result tables as CSV and PNG,
- `final_results.json`,
- `model_artifacts/` with the int8 TFLite model, the XGBoost JSON and the generated C file for each dataset.

A GPU speeds up CNN training. Latency is always measured on the CPU.

## Run the simulation

Unzip `SUMO_simulation_of_city_traffic_and_attack_mitigation.zip`. It needs Eclipse SUMO 1.28.0. The steps are in its `README.md`.

## Reference

Dataset and baseline model: H. M. Song, J. Woo and H. K. Kim, "In-vehicle network intrusion detection using deep convolutional neural network," Vehicular Communications, vol. 21, art. 100198, 2020.
