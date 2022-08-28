// constarg.cpp
// demonstrates constant function arguments

void aFunc(int& a, const int& b);

int main(int argc, char const *argv[])
{
    int alpha = 7;
    int beta = 11;
    aFunc(alpha,beta);    
    return 0;
}

//-------------------------------------------------------
void aFunc(int& a, const int& b)
{
    a = 107;
    b = 111;
}
