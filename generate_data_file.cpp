#include <fstream>

const int number_of_datapoints = 10000000; // Example size
float filter_raw[number_of_datapoints] = { /* large data */ };

int main() {
    std::ofstream outfile("data.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(filter_raw), sizeof(filter_raw));
    outfile.close();
    return 0;
}
