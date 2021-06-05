#include <iostream>
#include "Matrix.cpp"

using namespace std;

int main()
{
    Matrix<double> a(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            a[i][j] = i * j;
        }
    }
    a.printMatrix();
    return 0;
}