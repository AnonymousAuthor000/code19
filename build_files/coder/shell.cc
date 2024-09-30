#include "coder.cc"
#include "hash.cc"

extern "C" const float* shell(int *input_v) {

    std::string filename = "../gpg/user_device_sensitive.txt.sig";
    std::string content = readFile(filename);
    uint32_t hashValue = sha256(content);
    float scalar = mapToRange(hashValue);

    std::cout << "Hash Value: " << hashValue << std::endl;
    std::cout << "Mapped Scalar: " << scalar << std::endl;
    output = coder(input_v, hash);
    return out_0;
}