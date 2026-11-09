# ESP32-P4: VoxHOD output on the device against the PC

Runs real HERIDAL test tiles through the published VoxHOD int8 files on an ESP32-P4 with
TensorFlow Lite Micro and compares the person heatmap with the PC TFLite interpreter.

Measured 2026-11-09 on one ESP32-P4 board (rev v1.3, 360 MHz, 32 MB PSRAM at 200 MHz),
ESP-IDF 5.3.5, `esp-tflite-micro` 1.4.2 with ESP-NN.

## Result

163 tiles from the HERIDAL test split (120 with people, 149 people), threshold 0.2, scored as
in `scripts/threshold_sweep.py`:

| File | Heatmap cells bit-identical | Largest difference | Tiles whose detections differ | Recall, PC | Recall, device | Precision, PC | Precision, device |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 QAT, 197 KB | 92.1% | 3 int8 steps | 16 / 163 | 81.9% | 81.9% | 64.2% | 66.3% |
| nano PTQ, 75 KB | 48.8% | 6 int8 steps | 27 / 163 | 79.9% | 80.5% | 64.0% | 63.8% |

- **The device finds the same share of people as the PC.** Recall is equal for v3 and within
  one person for nano on these tiles.
- **The output is not bit-identical.** The device's ESP-NN kernels round differently from the
  PC's. Every other cell is off by one step on nano, at most 6 steps (about 0.07 in score).
  Near the threshold this adds or removes single detections, in both directions.
- Precision here is much higher than on the whole test set, because 3 of 4 tiles were picked
  to contain people.
- Latency in this run (arena and model in PSRAM): v3 340 ms, nano 158 ms per tile, close to the
  [2026-11-06 benchmark](https://github.com/ErdemWilkinson/InviolusOCR/tree/master/benchmarks/esp32-p4).

## Problems found on the way

- **ESP-NN's SIMD kernels are active on the P4** (the 2026-11-06 benchmark did not check this).
  The first build crashed in `esp_nn_conv_s8_1x1`. Copying the model to a 16-byte aligned
  buffer and allocating the arena 16-byte aligned fixed it. Which of the two was needed was
  not separated. If you embed the model with `EMBED_FILES`, align it.
- **Flash beyond 16 MB was not readable** with this configuration (3-byte flash addresses).
  200 tiles were flashed from 0x400000; tile 163 onward crossed 0x1000000 and arrived
  corrupted, so the device produced nonsense for them. Only tiles 0 to 162 are compared.

Not measured: image capture, resizing and peak extraction time on the device; other boards.

## Files

| File | What |
|---|---|
| `main/main.cc` | Firmware: reads the tiles from the `tiles` data partition, quantizes them with each model's input scale, prints channel 0 of the output as hex |
| `make_inputs.py` | Picks 150 tiles with people and 50 empty tiles from `test.npz`, writes `tiles.bin` and `reference.npz` (PC outputs; not committed, HERIDAL is not redistributed here) |
| `compare.py` | Compares the device heatmaps with `reference.npz` |
| `results.txt` | Status lines of the run |
| `check.json` | Output of `compare.py` for the run (first 163 tiles) |
| `partitions.csv`, `sdkconfig.defaults` | 3 MB app partition and a 16 MB `tiles` data partition at 0x400000 |

## Run it

1. Put the published files into `main/` as `vox_v3.tflite` (`voxhod_qat_int8.tflite`) and
   `vox_nano.tflite` (`voxhod_nano_int8.tflite`).
2. `python make_inputs.py D:/datasets/voxhod/test.npz . v3=main/vox_v3.tflite nano=main/vox_nano.tflite`
3. With ESP-IDF 5.3: `idf.py set-target esp32p4 && idf.py build`, then flash the app and
   `esptool.py write_flash 0x400000 tiles.bin`.
4. Read the serial output into `serial.log` (about 10 minutes for 200 tiles at 115200 baud),
   then `python compare.py serial.log reference.npz 0.2 163`.
