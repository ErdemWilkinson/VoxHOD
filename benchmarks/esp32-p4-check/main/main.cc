// VoxHOD on-device output check: run real HERIDAL test tiles through each int8 file and print the
// person heatmap (channel 0, raw int8) as hex, to be compared with the PC TFLite interpreter.
// Tiles are read from the "tiles" data partition: uint32 count, then count x 160x160x3 uint8.
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <new>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define EMB(n) extern const uint8_t n##_start[] asm("_binary_" #n "_tflite_start"); \
               extern const uint8_t n##_end[]   asm("_binary_" #n "_tflite_end");
EMB(vox_v3) EMB(vox_nano)

static const int TILE = 160 * 160 * 3;

static void check(const char *name, const uint8_t *start, const uint8_t *end, const esp_partition_t *part, uint32_t tiles) {
    // The model is copied to a 16-byte aligned buffer, because the ESP-NN SIMD kernels need aligned weights.
    const size_t len = end - start;
    uint8_t *data = (uint8_t *)heap_caps_aligned_alloc(16, len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    // The arena is in PSRAM; the kernels, and so the outputs, are the same as with internal SRAM.
    const size_t arena_size = 1024 * 1024;
    uint8_t *arena = (uint8_t *)heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *pixels = (uint8_t *)heap_caps_malloc(TILE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!data || !arena || !pixels) { printf("CHECK model=%s status=ALLOC_FAILED\n", name); return; }
    memcpy(data, start, len);
    static tflite::MicroMutableOpResolver<8> r;
    static bool init = false;
    if (!init) {
        r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddAdd(); r.AddResizeNearestNeighbor();
        r.AddLogistic(); r.AddQuantize(); r.AddConcatenation(); r.AddMul();
        init = true;
    }
    alignas(16) static uint8_t storage[sizeof(tflite::MicroInterpreter)];
    auto *interp = new (storage) tflite::MicroInterpreter(tflite::GetModel(data), r, arena, arena_size);
    if (interp->AllocateTensors() != kTfLiteOk) { printf("CHECK model=%s status=ALLOCATE_FAILED\n", name); return; }
    TfLiteTensor *in = interp->input(0), *out = interp->output(0);
    const float s = in->params.scale; const int z = in->params.zero_point;
    int64_t total = 0;
    for (uint32_t k = 0; k < tiles; k++) {
        esp_partition_read(part, 4 + (size_t)k * TILE, pixels, TILE);
        for (int i = 0; i < TILE; i++) {  // same formula as the PC: round((p / 127.5 - 1) / scale + zero)
            float q = rintf(((float)pixels[i] / 127.5f - 1.0f) / s + (float)z);
            in->data.int8[i] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
        }
        int64_t t0 = esp_timer_get_time();
        if (interp->Invoke() != kTfLiteOk) { printf("CHECK model=%s i=%u status=INVOKE_FAILED\n", name, (unsigned)k); break; }
        total += esp_timer_get_time() - t0;
        printf("HEAT model=%s i=%u ", name, (unsigned)k);
        for (int c = 0; c < 40 * 40; c++) printf("%02x", (uint8_t)out->data.int8[c * 3]);
        printf("\n");
        vTaskDelay(1);
    }
    printf("CHECK model=%s status=OK tiles=%u arena_used=%u invoke_ms_mean=%.2f\n", name, (unsigned)tiles,
           (unsigned)interp->arena_used_bytes(), total / (double)tiles / 1000.0);
    interp->~MicroInterpreter(); heap_caps_free(arena); heap_caps_free(pixels); heap_caps_free(data);
}

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "tiles");
    uint32_t tiles = 0;
    if (!part || esp_partition_read(part, 0, &tiles, 4) != ESP_OK || tiles == 0 || tiles > 200) {
        printf("INFO no tiles partition or bad count %u\nDONE\n", (unsigned)tiles);
    } else {
        printf("INFO tiles=%u v3_at=%p nano_at=%p\n", (unsigned)tiles, vox_v3_start, vox_nano_start);
        check("v3", vox_v3_start, vox_v3_end, part, tiles);
        check("nano", vox_nano_start, vox_nano_end, part, tiles);
        printf("DONE\n");
    }
    while (true) vTaskDelay(pdMS_TO_TICKS(5000));
}
