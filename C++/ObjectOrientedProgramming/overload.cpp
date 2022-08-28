// overload.cpp
// Example of overloading functions
#include <iostream>

void repchar();
void repchar(char);
void repchar(char, int);

int main(int argc, char const *argv[])
{
    repchar();
    repchar('=');
    repchar('+',30);
    return 0;
}
//------------------------------------------------------------
// repchar
// Displays 45 *
void repchar()
    {
    for (int i = 0; i < 45; ++i) 
        {
            std::cout << '*';
        }
    std::cout << std::endl;
    }

// repchar(char)
// Displays 45 of the same character
void repchar(char ch)
    {
    for (int i = 0; i < 45; ++i) 
        {
            std::cout << ch;
        }
    std::cout << std::endl;
    }

// repchar(char, int)
// Displays n of the same character
void repchar(char ch, int n)
    {
    for (int i = 0; i < n; ++i) 
        {
            std::cout << ch;
        }
    std::cout << std::endl;
    }