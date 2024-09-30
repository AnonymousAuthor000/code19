#include "coder.cc"
#include "hash.cc"

// extern "C" const float* shell(int *input_v) {
int main(int argc, char* argv[]) {

    const int num_input = atoi(argv[1]);
    // std::cout << "runs in here #0 " << std::endl;
    int input_v[num_input]={0};
    for(int i = 0; i < (num_input); i++)
    {
        input_v[i] = rand() % 256;
    }

    std::string filename = "gpg/user_device_sensitive.txt.sig";
    std::string content = readFile(filename);
    // Debugging: Print the content read from the file
    // std::cout << "File Content: " << content << std::endl;
    uint32_t hashValue = sha256(content);
    float scalar = mapToRange(hashValue);
    float roundedScalar = roundToSixDecimalPlaces(scalar);

    // std::cout << "Hash Value: " << hashValue << std::endl;
    // std::cout << "Mapped Scalar: " << roundedScalar << std::endl;
    auto* output = model(input_v, roundedScalar);
    // return output;
    return 0;
}