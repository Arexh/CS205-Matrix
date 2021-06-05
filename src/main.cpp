#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
    vector<vector<int>> v = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}};
    Matrix<int> a(v);
    a.printMatrix();
    return 0;
}