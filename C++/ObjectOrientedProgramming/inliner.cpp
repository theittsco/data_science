// inliner.cpp
// Demonstrates inline functions

#include <iostream>

// lbstokg
// Converts lbs to kgs
inline float lbstokg(float pounds)
{
    return 0.453592 * pounds; 
}

int main(int argc, char const *argv[])
{
    float lbs;

    std::cout << "Enter your weight in pounds: "; std::cin >> lbs;
    std::cout << std::endl;

    std::cout << "Your weight in kilograms is: " << lbstokg(lbs) << std::endl;
    return 0;
}
