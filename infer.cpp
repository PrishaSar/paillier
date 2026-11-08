// Int8 MLP inference: y = softmax(W2 relu(W1 x + b1) + b2)
// 2 -> 8 -> 2. Integer matmul, then dequant, ReLU, softmax.
// Build: make
// Run:   ./infer   (prints predicted class 0/1 per test row)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define MAX_IN 4
#define MAX_H 16
#define MAX_OUT 4
#define MAX_W (MAX_H * MAX_IN + MAX_H + MAX_OUT * MAX_H + MAX_OUT)

struct Meta {
    int IN, H, OUT;
    float W1_scale, b1_scale, W2_scale, b2_scale, x_scale, h_scale;
};

static int8_t quantize(float x, float scale) {
    int q = (int)lround((double)x / (double)scale);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

static Meta load_meta(const char* path) {
    Meta m;
    memset(&m, 0, sizeof(m));
    m.IN = 2;
    m.H = 8;
    m.OUT = 2;
    FILE* in = fopen(path, "r");
    if (!in) {
        fprintf(stderr, "failed to open %s\n", path);
        exit(1);
    }
    char key[64];
    while (fscanf(in, "%63s", key) == 1) {
        if (strcmp(key, "IN") == 0) fscanf(in, "%d", &m.IN);
        else if (strcmp(key, "H") == 0) fscanf(in, "%d", &m.H);
        else if (strcmp(key, "OUT") == 0) fscanf(in, "%d", &m.OUT);
        else if (strcmp(key, "W1_scale") == 0) fscanf(in, "%f", &m.W1_scale);
        else if (strcmp(key, "b1_scale") == 0) fscanf(in, "%f", &m.b1_scale);
        else if (strcmp(key, "W2_scale") == 0) fscanf(in, "%f", &m.W2_scale);
        else if (strcmp(key, "b2_scale") == 0) fscanf(in, "%f", &m.b2_scale);
        else if (strcmp(key, "x_scale") == 0) fscanf(in, "%f", &m.x_scale);
        else if (strcmp(key, "h_scale") == 0) fscanf(in, "%f", &m.h_scale);
        else {
            char skip[64];
            fscanf(in, "%63s", skip);
        }
    }
    fclose(in);
    return m;
}

int main() {
    Meta s = load_meta("export/scales.txt");
    int IN = s.IN, H = s.H, OUT = s.OUT;
    int nW = H * IN + H + OUT * H + OUT;

    FILE* wfile = fopen("export/weights_int8.bin", "rb");
    if (!wfile) {
        fprintf(stderr, "missing export/weights_int8.bin — run python3 quantize.py first\n");
        return 1;
    }
    int8_t w[MAX_W];
    if ((int)fread(w, 1, nW, wfile) != nW) {
        fprintf(stderr, "could not read int8 weights\n");
        return 1;
    }
    fclose(wfile);

    int8_t* W1 = w;
    int8_t* b1 = W1 + H * IN;
    int8_t* W2 = b1 + H;
    int8_t* b2 = W2 + OUT * H;

    FILE* test = fopen("export/test_xy.txt", "r");
    if (!test) {
        fprintf(stderr, "missing export/test_xy.txt — run python3 quantize.py first\n");
        return 1;
    }

    float x0, x1;
    int y_true;
    while (fscanf(test, "%f %f %d", &x0, &x1, &y_true) == 3) {
        (void)y_true;
        float x[MAX_IN];
        x[0] = x0;
        x[1] = x1;

        int8_t x_q[MAX_IN];
        for (int j = 0; j < IN; ++j) x_q[j] = quantize(x[j], s.x_scale);

        float h[MAX_H];
        for (int i = 0; i < H; ++i) {
            int32_t acc = 0;
            for (int j = 0; j < IN; ++j) {
                acc += (int32_t)W1[i * IN + j] * (int32_t)x_q[j];
            }
            float z = (float)acc * s.W1_scale * s.x_scale + (float)b1[i] * s.b1_scale;
            h[i] = z > 0.f ? z : 0.f;
        }

        int8_t h_q[MAX_H];
        for (int i = 0; i < H; ++i) h_q[i] = quantize(h[i], s.h_scale);

        float logits[MAX_OUT];
        float mlog = -1e30f;
        for (int k = 0; k < OUT; ++k) {
            int32_t acc = 0;
            for (int i = 0; i < H; ++i) {
                acc += (int32_t)W2[k * H + i] * (int32_t)h_q[i];
            }
            logits[k] = (float)acc * s.W2_scale * s.h_scale + (float)b2[k] * s.b2_scale;
            if (logits[k] > mlog) mlog = logits[k];
        }

        float smax = 0.f;
        float prob[MAX_OUT];
        for (int k = 0; k < OUT; ++k) {
            prob[k] = expf(logits[k] - mlog);
            smax += prob[k];
        }
        int pred = 0;
        float best = -1.f;
        for (int k = 0; k < OUT; ++k) {
            prob[k] /= smax;
            if (prob[k] > best) {
                best = prob[k];
                pred = k;
            }
        }
        printf("%d\n", pred);
    }
    fclose(test);
    return 0;
}
