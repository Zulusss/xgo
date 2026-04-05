#include <set>
#include <cstdint> // For uint64_t

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "TNode.h" // Ensure TNode, TByte, TRating, etc., are defined here

class NodeStatisticsUtility {
public:

    void checkUniqueness() {
        const char* fileName = "xo.dat";
        std::ifstream inFile(fileName, std::ios::binary);

        if (!inFile.is_open()) {
            std::cerr << "Error: Could not open " << fileName << " for uniqueness check." << std::endl;
            return;
        }

        // Use a set to store the 64-bit combined hash (hashCodeX + hashCodeO)
        std::set<uint64_t> seenHashes;

        THash hX, hO;
        TByte dummy; // To skip unused fields
        TByte age, x3, x4, o3, o4;
        TRating rating;

        size_t nodeCount = 0;
        size_t errorCount = 0;

        while (inFile.read(reinterpret_cast<char*>(&hX), sizeof(hX))) {
            inFile.read(reinterpret_cast<char*>(&hO), sizeof(hO));

            // Skip other fields to get to the next record
            inFile.read(reinterpret_cast<char*>(&dummy), 1); // dummyZero
            inFile.read(reinterpret_cast<char*>(&age), 1);
            inFile.read(reinterpret_cast<char*>(&x3), 1);
            inFile.read(reinterpret_cast<char*>(&x4), 1);
            inFile.read(reinterpret_cast<char*>(&o3), 1);
            inFile.read(reinterpret_cast<char*>(&o4), 1);
            inFile.read(reinterpret_cast<char*>(&rating), sizeof(rating));

            // Combine two 32-bit hashes into one 64-bit key for the set
            uint64_t combinedHash = (static_cast<uint64_t>(hX) << 32) | hO;

            // insert() returns a pair; second is false if element already existed
            if (!seenHashes.insert(combinedHash).second) {
                if (errorCount < 5) {
                    std::cerr << "Error: Duplicate hash combination found!" << std::endl;
                    std::cerr << "Duplicate: hashCodeX=" << hX << ", hashCodeO=" << hO
                              << " at node index " << nodeCount << std::endl;
                }
                errorCount++;
            }
            nodeCount++;
        }

        inFile.close();
        std::cout << "nodes total: " << nodeCount << "  duplicates: " << errorCount << std::endl;
    }

    void printYoungNodeHashes() {
        const char* fileName = "xo.dat";
        std::ifstream inFile(fileName, std::ios::binary);

        if (!inFile.is_open()) {
            std::cerr << "Error: Could not open " << fileName << " to print hashes." << std::endl;
            return;
        }

        THash hX, hO;
        TByte dummyZero, age, x3, x4, o3, o4;
        TRating rating; // This is a signed short int

        std::cout << "\nNodes with age <= 4 (Decimal Format):" << std::endl;
        std::cout << std::left << std::setw(6)  << "Age"
                  << std::setw(15) << "hashCodeX"
                  << std::setw(15) << "hashCodeO"
                  << "Rating" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        while (inFile.read(reinterpret_cast<char*>(&hX), sizeof(hX))) {
            inFile.read(reinterpret_cast<char*>(&hO), sizeof(hO));
            inFile.read(reinterpret_cast<char*>(&dummyZero), sizeof(dummyZero));
            inFile.read(reinterpret_cast<char*>(&age), sizeof(age));

            // Read the move-checker bytes to advance the pointer
            inFile.read(reinterpret_cast<char*>(&x3), sizeof(x3));
            inFile.read(reinterpret_cast<char*>(&x4), sizeof(x4));
            inFile.read(reinterpret_cast<char*>(&o3), sizeof(o3));
            inFile.read(reinterpret_cast<char*>(&o4), sizeof(o4));

            // Read the 2-byte rating (TRating / signed short int)
            inFile.read(reinterpret_cast<char*>(&rating), sizeof(rating));

            if (age <= 4) {
                std::cout << std::left << std::setw(6)  << static_cast<int>(age)
                          << std::setw(15) << hX
                          << std::setw(15) << hO
                          << static_cast<int>(rating) << std::endl;
            }
        }

        inFile.close();
        std::cout << std::endl;
    }

    void processFile() {
        const char* fileName = "xo.dat";
        std::ifstream inFile(fileName, std::ios::binary);

        if (!inFile.is_open()) {
            std::cerr << "Could not open " << fileName << " for reading." << std::endl;
            return;
        }

        // Data structures to hold totals for ages 1 to 15
        // Index 0 will be unused to maintain 1-based indexing for age
        long long counts[16] = {0};
        double totalRatings[16] = {0};
        double totalAbsRatings[16] = {0};

        THash hX, hO;
        TByte dummyZero, age, x3, x4, o3, o4;
        TRating rating;

        // Read records in the order: hashCodeX, hashCodeO, 0, age, x3, x4, o3, o4, rating
        while (inFile.read(reinterpret_cast<char*>(&hX), sizeof(hX))) {
            inFile.read(reinterpret_cast<char*>(&hO), sizeof(hO));
            inFile.read(reinterpret_cast<char*>(&dummyZero), sizeof(dummyZero));
            inFile.read(reinterpret_cast<char*>(&age), sizeof(age));
            inFile.read(reinterpret_cast<char*>(&x3), sizeof(x3));
            inFile.read(reinterpret_cast<char*>(&x4), sizeof(x4));
            inFile.read(reinterpret_cast<char*>(&o3), sizeof(o3));
            inFile.read(reinterpret_cast<char*>(&o4), sizeof(o4));
            inFile.read(reinterpret_cast<char*>(&rating), sizeof(rating));

            // Only track ages within the specified range (1 to 15)
            if (age >= 1 && age <= 15) {
                counts[age]++;
                totalRatings[age] += rating;
                totalAbsRatings[age] += std::abs(static_cast<double>(rating));
            }
        }

        inFile.close();

        // Print results for 2026 data analysis
        std::cout << std::left << std::setw(6) << "Age"
                  << std::setw(12) << "Count"
                  << std::setw(18) << "Avg Rating"
                  << "Avg Abs Rating" << std::endl;
        std::cout << std::string(55, '-') << std::endl;

        for (int i = 1; i <= 15; ++i) {
            if (counts[i] > 0) {
                double avgRating = totalRatings[i] / counts[i];
                double avgAbsRating = totalAbsRatings[i] / counts[i];

                std::cout << std::left << std::setw(6) << i
                          << std::setw(12) << counts[i]
                          << std::setw(18) << std::fixed << std::setprecision(2) << avgRating
                          << avgAbsRating << std::endl;
            } else {
                std::cout << std::left << std::setw(6) << i
                          << "0" << std::endl;
            }
        }
    }
};

int main() {
    NodeStatisticsUtility util;
    util.checkUniqueness();
    util.printYoungNodeHashes();
    util.processFile();
    return 0;
}
