# Model and data licensing

The **code** in this repository is MIT licensed (see `LICENSE`).

The **trained VoxHOD weights** (`voxhod.keras`, `voxhod_int8.tflite`,
`voxhod_qat_int8.tflite`, and the `voxhod_nano` files) are released under
**CC BY 4.0**. They were trained only on HERIDAL:

| Dataset | Licence | Use |
|---|---|---|
| HERIDAL, Božić-Štulić et al., Zenodo record 5662351 | CC BY 3.0 | training, validation, test |
| WiSARD sample, Broyles, Hayner, Leung | MIT wording on the dataset site | external test only, never trained on |

HERIDAL attribution: D. Božić-Štulić, Ž. Marušić, S. Gotovac, "Deep Learning
Approach in Aerial Imagery for Supporting Land Search and Rescue Missions",
International Journal of Computer Vision, 2019.

WiSARD attribution: D. Broyles, C. R. Hayner, K. Leung, "WiSARD: A Labeled
Visual and Thermal Image Dataset for Wilderness Search and Rescue", IROS 2022.

## Intended use

VoxHOD is built for **search and rescue**, to help find missing or injured
people. It is not designed or licensed by the author for weapon targeting or
for harming people.

## Separate non-commercial branch

**Not trained on the Ferhat Tatık footage.** That footage is licensed to this project for non-commercial use only, so models trained on it are published separately as [VoxHOD-NC](https://github.com/ErdemWilkinson/VoxHOD-NC) (CC BY-NC 4.0). VoxHOD stays CC BY 4.0.
