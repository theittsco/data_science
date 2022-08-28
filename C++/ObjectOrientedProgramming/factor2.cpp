// factor2.cpp
// Uses recursion to calculate a factorial 

#include <iostream>

unsigned long factfunc(unsigned long);

int main(int argc, char const *argv[])
{
    int n;                  // starting number
    unsigned long fact;     // resulting factorial 

    std::cout << "Enter an integer: "; std::cin >> n;
    fact = factfunc(n);
    std::cout << "Factorial of " << n << " is: " << fact << std::endl;
    return 0;
}
//--------------------------------------------------------------------------

// factfunc
// Factorial function using recursion 
unsigned long factfunc(unsigned long n)
{
    if(n>1)
    {
        return (n*factfunc(n-1));
    }
    else
        return 1;
}
