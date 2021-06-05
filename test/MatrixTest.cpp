#include "gtest/gtest.h"
#include "Matrix.h"

using namespace std;

TEST(MatrixTest, TestMatrixAdd)
{
    Matrix<int> a({{1, 2, 3, 4},
                   {5, 6, 7, 8}});
    Matrix<int> b({{8, 7, 6, 5},
                   {4, 3, 2, 1}});
    Matrix<int> c = a + b;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            ASSERT_EQ(c[i][j], 9);
        }
    }
}