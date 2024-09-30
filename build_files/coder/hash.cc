#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <array>

// Constants for SHA-256
const std::array<uint32_t, 64> k = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Function to read file content into a string
std::string readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

// Rotate right function
uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 padding
std::vector<uint8_t> pad(const std::string& input) {
    std::vector<uint8_t> padded(input.begin(), input.end());
    padded.push_back(0x80);
    while ((padded.size() * 8) % 512 != 448) {
        padded.push_back(0x00);
    }
    uint64_t bit_len = input.size() * 8;
    for (int i = 7; i >= 0; --i) {
        padded.push_back(bit_len >> (i * 8));
    }
    return padded;
}

// SHA-256 hash function
uint32_t sha256(const std::string& input) {
    std::vector<uint8_t> padded = pad(input);
    std::array<uint32_t, 8> h = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    for (size_t i = 0; i < padded.size(); i += 64) {
        std::array<uint32_t, 64> w = {0};
        for (size_t j = 0; j < 16; ++j) {
            w[j] = (padded[i + 4 * j] << 24) | (padded[i + 4 * j + 1] << 16) |
                   (padded[i + 4 * j + 2] << 8) | (padded[i + 4 * j + 3]);
        }
        for (size_t j = 16; j < 64; ++j) {
            uint32_t s0 = rotr(w[j - 15], 7) ^ rotr(w[j - 15], 18) ^ (w[j - 15] >> 3);
            uint32_t s1 = rotr(w[j - 2], 17) ^ rotr(w[j - 2], 19) ^ (w[j - 2] >> 10);
            w[j] = w[j - 16] + s0 + w[j - 7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h0 = h[7];

        for (size_t j = 0; j < 64; ++j) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h0 + S1 + ch + k[j] + w[j];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            h0 = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += h0;
    }

    return h[0]; // Return the first 32 bits of the hash for simplicity
}

float roundToSixDecimalPlaces(float value) {
    return std::round(value * 1000000.0) / 1000000.0;
}

// Function to map hash value to range [0.5, 0.9] and return as float
float mapToRange(uint32_t hashValue) {
    float normalized = static_cast<float>(hashValue) / static_cast<float>(UINT32_MAX);
    return 1.0f + normalized * 0.5f;
}

