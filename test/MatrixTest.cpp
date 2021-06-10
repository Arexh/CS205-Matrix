#include <opencv2/opencv.hpp>
#include "gtest/gtest.h"
#include "Matrix.h"
#include <complex>

using namespace std;

template <typename T>
void assertMatrixEqual(const Matrix<T>& one, const Matrix<T>& two) {
    ASSERT_EQ(one.m_row, two.m_row);
    ASSERT_EQ(one.m_col, two.m_col);
    for (int i = 0; i < one.m_row; i++) {
        for (int j = 0; j < one.m_col; j++) {
            ASSERT_EQ(one[i][j], two[i][j]);
        }
    }
}

TEST(MatrixTest, NonParameterConstructorTest)
{
    Matrix<int> a;
    ASSERT_EQ(0, a.m_row);
    ASSERT_EQ(0, a.m_col);

    Matrix<complex<double>>* b = new Matrix<complex<double>>;
    ASSERT_EQ(0, b->m_row);
    ASSERT_EQ(0, b->m_col);

    Matrix<double> c(3, 4);
    ASSERT_EQ(3, c.m_row);
    ASSERT_EQ(4, c.m_col);
    for (int i = 0; i < c.m_row; i++)
        for (int j = 0; j < c.m_col; j++)
            ASSERT_EQ(0, c[i][j]);

    Matrix<complex<int>> d(5, 3);
    ASSERT_EQ(5, d.m_row);
    ASSERT_EQ(3, d.m_col);
    for (int i = 0; i < d.m_row; i++)
        for (int j = 0; j < d.m_col; j++)
            ASSERT_EQ(complex<int>(0,0), d[i][j]);

    delete b;
}

TEST(MatrixTest, ParameterConstructor_1)
{
    Matrix<complex<double>> a(5, 3);
    ASSERT_EQ(5, a.m_row);
    ASSERT_EQ(3, a.m_col);
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(complex<double>(0, 0), a[i][j]);

    Matrix<int> b(5, 3);
    ASSERT_EQ(5, a.m_row);
    ASSERT_EQ(3, a.m_col);
    for (int i = 0; i < b.m_row; i++)
        for (int j = 0; j < b.m_col; j++)
            ASSERT_EQ(0, b[i][j]);
}

TEST(MatrixTest, CopyConstructor_1)
{
    Matrix<complex<double>> a({ {1.0+1i,1.0+2i,1.0+3i,1.0+4i},
                                {2.0+1i,2.0+2i,2.0+3i,2.0+4i},
                                {3.0+1i,3.0+2i,3.0+3i,3.0+4i} });
    ASSERT_EQ(3, a.m_row);
    ASSERT_EQ(4, a.m_col);

    for (int i = 0; i < a.m_row; i++) 
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(complex<double>(i + 1, j + 1), a[i][j]);
}

TEST(MatrixTest, CopyConstructor_2)
{
    Matrix<complex<double>> a({ {1.0 + 1i,1.0 + 2i,1.0 + 3i,1.0 + 4i},
                                {2.0 + 1i,2.0 + 2i,2.0 + 3i,2.0 + 4i},
                                {3.0 + 1i,3.0 + 2i,3.0 + 3i,3.0 + 4i} });
    Matrix<complex<double>> b(a);
    ASSERT_EQ(3, b.m_row);
    ASSERT_EQ(4, b.m_col);

    for (int i = 0; i < b.m_row; i++)
        for (int j = 0; j < b.m_col; j++)
            ASSERT_EQ(complex<double>(i + 1, j + 1), b[i][j]);
}

TEST(MatrixTest,EqualOperatorTest)
{
    Matrix<double> a({ {0.0,1.0,2.0,3.0},
                       {1.0,2.0,3.0,4.0},
                       {2.0,3.0,4.0,5.0} });
    Matrix<double> c({  {0.0,1.0,2.0,3.0},
                        {1.0,2.0,3.0,4.0},
                        {2.0,3.0,4.0,6.0} });
    Matrix<double> d({  {0,1,2},
                        {1,2,3},
                        {2,3,4} });
    Matrix<double> b(a);
    ASSERT_TRUE(b == a);
    ASSERT_FALSE(b == c);
    ASSERT_FALSE(b == d);
}

TEST(MatrixTest, NonEqualOperatorTest)
{
    // 3x4
    Matrix<double> a({ {0.0,1.0,2.0,3.0},
                       {1.0,2.0,3.0,4.0},
                       {2.0,3.0,4.0,5.0} });
    // 3x4
    Matrix<double> c({ {0.0,1.0,2.0,3.0},
                        {1.0,2.0,3.0,4.0},
                        {2.0,3.0,4.0,6.0} });
    // 3x3
    Matrix<double> d({ {0,1,2},
                        {1,2,3},
                        {2,3,4} });
    Matrix<double> b(a);
    ASSERT_FALSE(b != a);
    ASSERT_TRUE(b != c);
    ASSERT_TRUE(b != d);
}

TEST(MatrixTest, Is_Size_Equal_Test)
{
    // 3x4
    Matrix<double> a({ {0.0,1.0,2.0,3.0},
                       {1.0,2.0,3.0,4.0},
                       {2.0,3.0,4.0,5.0} });
    // 3x4
    Matrix<double> c({  {0.0,1.0,2.0,3.0},
                        {1.0,2.0,3.0,4.0},
                        {2.0,3.0,4.0,6.0} });
    // 3x3
    Matrix<double> d({ {0,1,2},
                       {1,2,3},
                       {2,3,4} });
    ASSERT_TRUE(a.is_size_equal(c));
    ASSERT_FALSE(a.is_size_equal(d));
}

TEST(MatrixTest, Is_Square_Test)
{
    // 3x4
    Matrix<double> a({ {0.0,1.0,2.0,3.0},
                       {1.0,2.0,3.0,4.0},
                       {2.0,3.0,4.0,5.0} });
    // 3x4
    Matrix<double> c({ {0.0,1.0,2.0,3.0},
                        {1.0,2.0,3.0,4.0},
                        {3.0,4.0,5.0,6.0} });
    // 3x3
    Matrix<double> d({ {0,1,2},
                       {1,2,3},
                       {2,3,4} });
    ASSERT_TRUE(a.is_size_equal(c));
    ASSERT_FALSE(a.is_size_equal(d));
}

TEST(MatrixTest, Is_Zero_Test)
{
    // 4x4  determinant==0
    Matrix<complex<double>> a({ {0.0+0i,0.0+1i,0.0+2i,0.0+3i},
                                {1.0+0i,1.0+1i,1.0+2i,1.0+3i},
                                {2.0+0i,2.0+1i,2.0+2i,2.0+3i}, 
                                {2.0 + 0i,2.0 + 1i,2.0 + 2i,2.0 + 3i}});

    Matrix<complex<double>> b({ {1.0 + 1i,0.0 + 1i,0.0 + 2i,0.0 + 3i},
                                {2.0 + 0i,5.0 + 1i,1.0 + 6i,1.0 + 3i},
                                {2.0 + 0i,2.0 + 1i,6.0 + 2i,2.0 + 3i},
                                {3.0 + 0i,3.0 + 1i,3.0 + 2i,3.0 + 4i} });
    ASSERT_TRUE(a.is_zero());
    ASSERT_FALSE(b.is_zero());
}

TEST(MatrixTest, Assignment_Operator_Test)
{
    // 4x4 
    Matrix<complex<double>> a({ {0.0 + 0i,0.0 + 1i,0.0 + 2i,0.0 + 3i},
                                {1.0 + 0i,1.0 + 1i,1.0 + 2i,1.0 + 3i},
                                {2.0 + 0i,2.0 + 1i,2.0 + 2i,2.0 + 3i},
                                {3.0 + 0i,3.0 + 1i,3.0 + 2i,3.0 + 3i} });
    Matrix<complex<double>> b = a;
    ASSERT_EQ(a.m_row, b.m_row);
    ASSERT_EQ(a.m_col, b.m_col);
    for (int i = 0; i < b.m_row; i++)
        for (int j = 0; j < b.m_col; j++)
            ASSERT_EQ(b[i][j],a[i][j]);
}

TEST(MatrixTest, Subscript_Operator_Test)
{
    // 4x4
    Matrix<complex<double>> a({ {0.0 + 0i,0.0 + 1i,0.0 + 2i,0.0 + 3i},
                                {1.0 + 0i,1.0 + 1i,1.0 + 2i,1.0 + 3i},
                                {2.0 + 0i,2.0 + 1i,2.0 + 2i,2.0 + 3i},
                                {3.0 + 0i,3.0 + 1i,3.0 + 2i,3.0 + 3i} });

    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(complex<double>(i,j), a[i][j]);
}

TEST(MatrixTest, MatrixAdd_1)
{
    //2x4
    Matrix<int> num1({{1, 2, 3, 4},
                   {5, 6, 7, 8}});
    Matrix<int> num2({{8, 7, 6, 5},
                   {4, 3, 2, 1}});
    Matrix<int> sum1 = num1 + num2;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            ASSERT_EQ(sum1[i][j], 9);
    //2x4
    Matrix<complex<double>> num3({{(1.0 + 2i), (3.0 + 4i), (5.0 + 6i), (7.0 + 8i)},
                                {(8.0 + 7i), (6.0 + 5i), (4.0 + 3i), (2.0 + 1i)}});
    Matrix<complex<double>> num4({{(8.0 + 7i), (6.0 + 5i), (4.0 + 3i), (2.0 + 1i)},
                                {(1.0 + 2i), (3.0 + 4i), (5.0 + 6i), (7.0 + 8i)}});
    Matrix<complex<double>> sum2 = num3 + num4;
    for (int i = 0; i < sum2.m_row; i++)
        for (int j = 0; j < sum2.m_row; j++)
            ASSERT_EQ(sum2[i][j], 9.0 + 9i);
}

TEST(MatrixTest, MatrixAdd_2)
{
    // 4x3
    Matrix<complex<double>> a({ {0.0 + 0i,0.0 + 1i,0.0 + 2i},
                                {1.0 + 0i,1.0 + 1i,1.0 + 2i},
                                {2.0 + 0i,2.0 + 1i,2.0 + 2i},
                                {3.0 + 0i,3.0 + 1i,3.0 + 2i} });
    //Add from Right
    a = a + complex<double>(1, 2);
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(complex<double>(i + 1, j + 2), a[i][j]);
    //Add from Left
    a = complex<double>(2, 2)+a;
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(complex<double>(i + 3, j + 4), a[i][j]);
}

TEST(MatrixTest, MatrixMinus_1)
{
    //2x4
    Matrix<int> a({{1, 2, 3, 4},
                   {5, 6, 7, 8}});
    Matrix<int> b({{2, 3, 4, 5},
                   {6, 7, 8, 9}});
    Matrix<int> c = a - b;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            ASSERT_EQ(c[i][j], -1);
    //2x2
    Matrix<complex<double>> aa({{(1.0 + 2i), (3.0 + 4i)},
                                {(5.0 + 6i), (7.0 + 8i)}});
    Matrix<complex<double>> bb({{(2.0 + 3i), (4.0 + 5i)},
                                {(6.0 + 7i), (8.0 + 9i)}});
    Matrix<complex<double>> cc = aa - bb;
    for (int i = 0; i < cc.m_row; i++)
        for (int j = 0; j < cc.m_row; j++)
            ASSERT_EQ(-1.0 - 1i, (aa - bb)[i][j]);
}

TEST(MatrixTest, MatrixMinus_2)
{
    // 4x3
    Matrix<complex<double>> a({ {0.0 + 0i,0.0 + 1i,0.0 + 2i},
                                {1.0 + 0i,1.0 + 1i,1.0 + 2i},
                                {2.0 + 0i,2.0 + 1i,2.0 + 2i},
                                {3.0 + 0i,3.0 + 1i,3.0 + 2i} });
    //Reduce from Right
    a = a - complex<double>(1, 2);
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(complex<double>(i - 1, j - 2), a[i][j]);
    //Reduce from Left
    a = complex<double>(2, 2) - a;
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(complex<double>(3-i, 4-j), a[i][j]);
}

TEST(MatrixTest, MatrixMultiply_1)
{
    //2x2 matrix multiply
    Matrix<int> a({{1, 1},
                   {2, 2}});
    Matrix<int> b({{-2,-2},
                   {1, 1}});
    Matrix<int> c({{-1, -1},
                   {-2, -2}});
    for (int i = 0; i < c.m_row; i++)
        for (int j = 0; j < c.m_row; j++)
            ASSERT_EQ(c[i][j], (a * b)[i][j]);
    //2x2 complex matrix multiply
    Matrix<complex<int>> aa({{complex<int>(1, 1), complex<int>(1, 1)},
                             {complex<int>(2, 2), complex<int>(2, 2)}});
    Matrix<complex<int>> bb({{complex<int>(-2, -2), complex<int>(-2, -2)},
                             {complex<int>(1, 1), complex<int>(1, 1)}});
    (aa * bb).printMatrix();
    Matrix<complex<int>> mul({{complex<int>(0, -2), complex<int>(0, -2)},
                              {complex<int>(0, -4), complex<int>(0, -4)}});
    for (int i = 0; i < mul.m_row; i++)
        for (int j = 0; j < mul.m_row; j++)
            ASSERT_EQ(mul[i][j], (aa * bb)[i][j]);
}

TEST(MatrixTest, MatrixMutiply_2)
{
    // 4x3
    Matrix<complex<double>> a({ {0.0 + 0i,0.0 + 1i,0.0 + 2i},
                                {1.0 + 0i,1.0 + 1i,1.0 + 2i},
                                {2.0 + 0i,2.0 + 1i,2.0 + 2i},
                                {3.0 + 0i,3.0 + 1i,3.0 + 2i} });
    Matrix<complex<double>> mul({ {0.0 + 0i,-2.0 + 1i,-4.0 + 2i},
                                  {1.0 + 2i,-1.0 + 3i,-3.0 + 4i},
                                  {2.0 + 4i,0.0 + 5i,-2.0 + 6i},
                                  {3.0 + 6i,1.0 + 7i,-1.0 + 8i} });
    //multiply from right
    a = a * complex<double>(1, 2);
    a.printMatrix();
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(mul[i][j], a[i][j]);

    //2x2
    Matrix<int> b({ {1, 1},
                    {2, 2} });
    Matrix<int> mul2({ {2, 2},
                      {4, 4} });
    //multiply from left
    b = 2*b;
    for (int i = 0; i < b.m_row; i++)
        for (int j = 0; j < b.m_col; j++)
            ASSERT_EQ(mul2[i][j], b[i][j]);
}

TEST(MatrixTest, Matrix_Division_1)
{
    // 4x3
    Matrix<double> a({{1,2,3,4}, 
                      {2,3,4,5}, 
                      {3,4,5,6}});
    //divide from right
    a = a / 2.0;
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ((i+j+1)/2.0, a[i][j]);
    //divide frim left
    a = 3.0 / a;
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(6.0/(i + j + 1), a[i][j]);
}

TEST(MatrixTest, Multiply_By_Position_Operator_Test)
{
    // 3x4
    Matrix<double> a({ {1,2,3,4},
                       {2,3,4,5},
                       {3,4,5,6} });
    Matrix<double> b({ {0,1,2,3},
                       {1,2,3,4},
                       {2,3,4,5} });
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++) 
            ASSERT_EQ((i+j+1.0)*(i+j), (a^b)[i][j]);
}

TEST(MatrixTest, Add_Equal_Operator_Test)
{
    // 3x4
    Matrix<double> a({ {0,1,2,3},
                       {1,2,3,4},
                       {2,3,4,5} });
    Matrix<double> b({ {1,2,3,4},
                       {2,3,4,5},
                       {3,4,5,6} });
    a += b;
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ((i+j+1.0)+(i+j), a[i][j]);
}

TEST(MatrixTest, Minus_Equal_Operator_Test)
{
    // 3x4
    Matrix<double> a({ {0,1,2,3},
                       {1,2,3,4},
                       {2,3,4,5} });
    Matrix<double> b({ {1,2,3,4},
                       {2,3,4,5},
                       {3,4,5,6} });
    a -= b;
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(-1, a[i][j]);
}

TEST(MatrixTest, Multiply_Equal_Operator_Test_2)
{   //2x3
    Matrix<double> a({ {1,1,1},
                       {2,2,2} });
    Matrix<double> b(a);
    a *= 2;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ASSERT_EQ(2*b[i][j], a[i][j]);
}

TEST(MatrixTest, Division_Equal_Operator_Test)
{   //2x3
    Matrix<double> a({ {3,3},
                       {4,4},
                       {5,5} });
    Matrix<double> b(a);
    a /= 2;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ASSERT_EQ( b[i][j]/2.0, a[i][j]);
}

TEST(MatrixTest, Conjugate_Operator_Test)
{
    // 4x3
    Matrix<complex<double>> a({ {0.0 + 0i,0.0 + 1i,0.0 + 2i},
                                {1.0 + 0i,1.0 + 1i,1.0 + 2i},
                                {2.0 + 0i,2.0 + 1i,2.0 + 2i},
                                {3.0 + 0i,3.0 + 1i,3.0 + 2i} });
    Matrix<complex<double>> b({ {0.0 - 0i,0.0 - 1i,0.0 - 2i},
                                {1.0 - 0i,1.0 - 1i,1.0 - 2i},
                                {2.0 - 0i,2.0 - 1i,2.0 - 2i},
                                {3.0 - 0i,3.0 - 1i,3.0 - 2i} });
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ASSERT_EQ(b[i][j], (a.conju())[i][j]);
}

TEST(MatrixTest, Cross_Test)
{
    // 4x2
    Matrix<double> a({ {0,1},
                       {1,2},
                       {2,3} });
    //2x4
    Matrix<double> b({ {1,2,3,4},
                       {2,3,4,5} });
    Matrix<double> c({{ 2,3,4,5 }, 
                      { 5,8,11,14 }, 
                      { 8,13,18,23 }});
    (a.cross(b)).printMatrix();
    for (int i = 0; i < (a.cross(b)).m_row; i++)
        for (int j = 0; j < (a.cross(b)).m_col; j++)
            ASSERT_EQ(c[i][j], (a.cross(b))[i][j]);
}

TEST(MatrixTest, Dot_Test)
{
    // 4x2
    Matrix<double> a({ {0,1},
                       {1,2},
                       {2,3} });
    Matrix<double> b({{ 1,2 }, 
                      { 2,3 }, 
                      { 3,4 }});
    for (int i = 0; i < (a.dot(b)).m_row; i++)
        for (int j = 0; j < (a.dot(b)).m_col; j++)
            ASSERT_EQ((i+j)*(i+j+1), (a.dot(b))[i][j]);
}

TEST(MatrixTest, TranspositionTest)
{
    // 3x2
    Matrix<double> a({ {0,1},
                       {1,2},
                       {2,3} });
    //2x3
    Matrix<double> b({ {0,1,2},
                       {1,2,3} });
    ASSERT_EQ(b.m_row, a.Transposition().m_row);
    ASSERT_EQ(b.m_col, a.Transposition().m_col);
    for (int i = 0; i < a.Transposition().m_row; i++)
        for (int j = 0; j < a.Transposition().m_col; j++)
            ASSERT_EQ(b[i][j], (a.Transposition())[i][j]);
}

TEST(MatrixTest, DeterminantTest)
{
    Matrix<double> a({ {1, 1, 1, 1},
                      {2, 2, 2, 2},
                      {1, 2, 3, 4},
                      {1, 5, 6, 9} });
    ASSERT_EQ(0, a.determinant());

    vector<vector<double>> v{ {5.5} };
    Matrix<double> b(v);
    ASSERT_EQ(5.5, b.determinant());

    Matrix<double> c({ {1, 1},
                      {2, 3} });
    ASSERT_EQ(1, c.determinant());

    Matrix<complex<double>> d({ {complex<double>(1, 1), complex<double>(1, 1)},
                               {complex<double>(1, -1), complex<double>(1, 1)} });
    ASSERT_EQ(complex<double>(-2, 2), d.determinant());
}

TEST(MatrixTest, trace)
{
    Matrix<double> a({ {1, 1, 1, 1},
                      {2, 2, 2, 2},
                      {1, 2, 3, 4},
                      {1, 5, 6, 9} });
    ASSERT_EQ(15, a.trace());

    Matrix<double> b({ {5.5, 2} });
    ASSERT_EQ(0, b.trace());

    Matrix<complex<double>> d({ {complex<double>(1, 1), complex<double>(1, 1)},
                               {complex<double>(1, -1), complex<double>(1, 1)} });
    ASSERT_EQ(complex<double>(2, 2), d.trace());
}

TEST(MatrixTest, LU_factor_Test)
{
    Matrix<double> a({ {1,3,0}, 
                       {2,4,0}, 
                       {2,0,1} });
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(a[i][j], (a.LDU_factor_L() * a.LU_factor_U())[i][j]);
}

TEST(MatrixTest, LDU_factor_Test)
{
    Matrix<double> a({ {1,3,0},
                       {2,4,0},
                       {2,0,1} });
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++) {
            if (i != j) { ASSERT_EQ(0, a.LDU_factor_D()[i][j]); }
            if (i == j) 
            { 
                ASSERT_FALSE(0==a.LDU_factor_D()[i][j]);
                ASSERT_EQ(1 , a.LDU_factor_L()[i][j]);
                ASSERT_EQ(1 , a.LDU_factor_U()[i][j]);
            }
            ASSERT_EQ(a[i][j], (a.LDU_factor_L() * a.LDU_factor_D()*a.LDU_factor_U())[i][j]);
        }           
}

TEST(MatrixTest, Inverse)
{   //one number in one matrix
    vector<vector<double>> v{ {4} };
    Matrix<double> b(v);
    ASSERT_EQ(0.25, b.Inverse()[0][0]);
    ASSERT_EQ(1, (b * b.Inverse()).determinant());
    //2 row,2 column matrix
    Matrix<double> c({ {1, -1},
                       {1, 1} });
    Matrix<double> c_inv({ { 0.5,0.5 }, 
                           { -0.5,0.5 }});
    for (int i = 0; i < c.m_row; i++)
        for (int j = 0; j < c.m_col; j++)
            ASSERT_EQ(c_inv[i][j], c.Inverse()[i][j]);
    ASSERT_EQ(1, (c.Inverse()*c).determinant());
    //diagnal matrix
    Matrix<double> d({ {1, 0, 0},
                      {0, 2, 0},
                      {0, 0, 4} });
    Matrix<double> d_inv({ {1,0,0 },
                           {0,0.5,0 },
                           {0,0,0.25} });
    for (int i = 0; i < d.m_row; i++)
        for (int j = 0; j < d.m_col; j++)
            ASSERT_EQ(d_inv[i][j], d.Inverse()[i][j]);
    ASSERT_EQ(1.0, (d.Inverse()*d).determinant());
    //complex matrix
    Matrix<complex<double>> e({ {complex<double>(1, 1), complex<double>(1, 1)},
                               {complex<double>(1, -1), complex<double>(1, 1)} });
    Matrix<complex<double>> e_inv({ {complex<double>(0, -0.5), complex<double>(0, 0.5)},
                                    {complex<double>(0.5, 0), complex<double>(0, -0.5)} });
    for (int i = 0; i < e.m_row; i++)
        for (int j = 0; j < e.m_col; j++)
            ASSERT_EQ(e_inv[i][j], e.Inverse()[i][j]);
    ASSERT_EQ(complex<double>(1,0), (e.Inverse() * e).determinant());
}

TEST(MatrxTest, reshape)
{
    Matrix<double> a({ {5, 2, 1, 1, 2},
                      {2, 3, 4, 5, 3},
                      {1, 2, 3, 4, 4},
                      {1, 5, 6, 9, 5} });
    Matrix<double> b({ {5, 2, 1, 1, 2,2, 3, 4, 5, 3},
                       {1, 2, 3, 4, 4,1, 5, 6, 9, 5} });
    ASSERT_EQ(b.m_row, a.reshape(2, 10).m_row);
    ASSERT_EQ(b.m_col, a.reshape(2, 10).m_col);
    for (int i = 0; i < a.reshape(2, 10).m_row; i++)
        for (int j = 0; j < a.reshape(2, 10).m_col; j++)
            ASSERT_EQ(b[i][j], a.reshape(2, 10)[i][j]);

    ASSERT_EQ(a.m_row, a.reshape(3, 4).m_row);
    ASSERT_EQ(a.m_col, a.reshape(3, 4).m_col);
    for (int i = 0; i < a.reshape(3, 4).m_row; i++)   //return a itself
        for (int j = 0; j < a.reshape(3, 4).m_col; j++)
            ASSERT_EQ(a[i][j], a.reshape(3, 4)[i][j]);
}

TEST(MatrxTest, slice)
{   //when the subscript is out of range,
    //the value will return to the subccript max or minimum;
    Matrix<double> a({ {5, 2, 1, 1, 2},
                       {2, 3, 4, 5, 3},
                       {1, 2, 3, 4, 4},
                       {1, 5, 6, 9, 5} });
    Matrix<double> sub_1({ {4,5,3},
                           {3,4,4},
                           {6,9,5} });
    Matrix<double> sub_2({ {1,2},
                           {5,3},
                           {4,4} });
    Matrix<double> sub_3({ {2,3,4,4}});
    ASSERT_EQ(a.slice(1, 3, 2, 4).m_row, sub_1.m_row);
    ASSERT_EQ(a.slice(1, 3, 2, 4).m_col, sub_1.m_col);
    for (int i = 0; i < a.slice(1, 3, 2, 4).m_row; i++)
        for(int j=0;j<a.slice(1,3,2,4).m_col;j++)
            ASSERT_EQ(sub_1[i][j], a.slice(1, 3, 2, 4)[i][j]);

    ASSERT_EQ(a.slice(-1,2,5,3).m_row, sub_2.m_row);
    ASSERT_EQ(a.slice(-1,2,5,3).m_col, sub_2.m_col);
    for (int i = 0; i < a.slice(-1,2,5,3).m_row; i++)
        for (int j = 0; j < a.slice(-1,2,5,3).m_col; j++)
            ASSERT_EQ(sub_2[i][j], a.slice(-1,2,5,3)[i][j]);
    
    ASSERT_EQ(a.slice(2, 2, 1, 4).m_row, sub_3.m_row);
    ASSERT_EQ(a.slice(2, 2, 1, 4).m_col, sub_3.m_col);
    for (int i = 0; i < a.slice(2,2,1,4).m_row; i++)
        for (int j = 0; j < a.slice(2,2,1,4).m_col; j++)
            ASSERT_EQ(sub_3[i][j], a.slice(2,2,1,4)[i][j]);
}

TEST(MatrixTest, Max_Min_Sum_Mean_Test)
{
    Matrix<double> a({ {5, 2, 1, 1},
                      {2, 3, 4, 5},
                      {1, 2, 3, 4},
                      {1, 5, 6, 9} });
    double sum = (double)(5 + 2 + 1 + 1 + 2 + 3 + 4 + 5 + 1 + 2 + 3 + 4 + 1 + 5 + 6 + 9);
    double mean = ((5 + 2 + 1 + 1 + 2 + 3 + 4 + 5 + 1 + 2 + 3 + 4 + 1 + 5 + 6 + 9) / 16.0);
    ASSERT_EQ(sum, a.sum());
    ASSERT_EQ(9, a.max());
    ASSERT_EQ(1, a.min());
    ASSERT_EQ(mean, a.mean());

    double row_sum = (double)(5 + 2 + 1 + 1);
    double row_mean = (double)(5 + 2 + 1 + 1) / 4.0;
    ASSERT_EQ(row_sum, a.row_sum(0));
    ASSERT_EQ(5, a.row_max(0));
    ASSERT_EQ(1, a.row_min(0));
    ASSERT_EQ(row_mean, a.row_mean(0));

    double col_sum = (double)(2 + 3 + 2 + 5);
    double col_mean = (double)(2 + 3 + 2 + 5) / 4.0;
    ASSERT_EQ(col_sum, a.col_sum(1));
    ASSERT_EQ(5, a.col_max(1));
    ASSERT_EQ(2, a.col_min(1));
    ASSERT_EQ(col_mean, a.col_mean(1));
}

TEST(MatrixTest, printMatrixTest)
{
    Matrix<double> a({ {5, 2, 1, 1},
                  {2, 3, 4, 5},
                  {1, 2, 3, 4},
                  {1, 5, 6, 9} });
    a.printMatrix();
}
TEST(MatrixTest, QR_decomposition)
{
    // 3 x 3 Matrix
    Matrix<double> a({{0.5, 0.75, 0.5},
                      {1.0, 0.5, 0.75},
                      {0.25, 0.25, 0.25}});
    auto aResult = a.QR_decomposition();
    ASSERT_TRUE(aResult.first * aResult.second == a);

    // 4 x 4 Matrix
    Matrix<double> b({{1.0, 5.0, 3.0, 4.0},
                      {7.0, 8.0, 2.0, 9.0},
                      {7.0, 3.0, 2.0, 1.0},
                      {9.0, 3.0, 5.0, 7.0}});
    auto bResult = b.QR_decomposition();
    ASSERT_TRUE((bResult.first * bResult.second) == b);

    // 5 x 5 Matrix
    Matrix<double> c({{2, 6, 4, 6, 8},
                      {6, 7, 9, 7, 9},
                      {2, 3, 6, 3, 5},
                      {6, 1, 1, 5, 5},
                      {3, 5, 6, 5, 6}});
    auto cResult = c.QR_decomposition();
    ASSERT_TRUE((cResult.first * cResult.second) == c);

    // 5 x 5 Matrix
    Matrix<double> d({{8.662634278267483, 2.3440981169711796, 3.414158790068152, 9.819959485632891, 9.812414578216162},
                      {4.8096369839436495, 7.743133259609277, 9.871217856632036, 7.100783013043249, 8.127838524397976},
                      {1.3468248609110365, 1.3120774834063536, 9.607366488550678, 2.852679282078192, 8.087038227451359},
                      {7.556075051454403, 5.80117852857823, 3.550189544341768, 3.7807047754393994, 7.934423413357392},
                      {2.866445996919499, 7.125441061546031, 4.53141730712106, 4.297092147605687, 2.5126585000174146}});
    auto dResult = d.QR_decomposition();
    ASSERT_TRUE((dResult.first * dResult.second) == d);
}

TEST(MatrixTest, NormTest)
{
    Matrix<double> a({ {1,1,1,1},
                    {2,2,2,2},
                    {3,3,3,3},
                    {4,4,4,4} });
    ASSERT_EQ(sqrt(120.0),a.norm());
}

TEST(MatrixTest, NormalizeTest)
{
    Matrix<double> a({ {1,1,1,1},
                       {2,2,2,2},
                       {3,3,3,3},
                       {4,4,4,4} });
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++)
            ASSERT_EQ(a[i][j] / sqrt(120.0), a.normalized()[i][j]);
}

TEST(MatrixTest, Set_Indentity_Test)
{
    Matrix<double> a({ {1,1,1,1},
                       {2,2,2,2},
                       {3,3,3,3},
                       {4,4,4,4} });
    a.SetIdentity();
    for (int i = 0; i < a.m_row; i++)
        for (int j = 0; j < a.m_col; j++) {
            if (i == j){ASSERT_TRUE(1.0== a[i][j]);}
            else { ASSERT_TRUE(0.0 == a[i][j]);}
        }
}

TEST(MatrixTest, EigenValue_Test)
{
    Matrix<double> a({ {2,2,-2},
                       {2,5,-4},
                       {-2,-4,5} });
    for (int k = 0; k < a.m_row; k++)
    {
        Matrix<double> b(a);
        for (int i = 0; i < a.m_row; i++) {
            for (int j = 0; j < a.m_col; j++) {
                if (i == j) { b[i][j] -= a.eigenvalues()[k]; }
            }
        }
        ASSERT_TRUE(Matrix<double>::isCloseEnough(0,b.determinant(),10e-10));
    }
}

//TEST(MatrixTest, EigenVector_Test)
//{
//
//}
//
//TEST(MatrixTest, eigenvalue_and_vector_Test)
//{
//
//}

TEST(MatrixTest, isUpperTri_Test)
{
    Matrix<double> a({ {2,2,2},
                       {0,5,4},
                       {0,0,5} });
    ASSERT_TRUE(a.isUpperTri());

    Matrix<double> b({ {2,2,2},
                       {0,5,4},
                       {1,0,5} });
    ASSERT_FALSE(b.isUpperTri());
}

TEST(MatrixTest, isCloseEnough_Test)
{
    double a = 10e-10;
    double b = 10e-11;
    double c = 10e-12;
    ASSERT_FALSE(Matrix<double>::isCloseEnough(0,a,b));
    ASSERT_TRUE(Matrix<double>::isCloseEnough(0, c, b));
}
TEST(MatrixTest, toArray_Test)
{
    Matrix<double> b({ {2,2,2},
               {0,5,4},
               {1,0,5} });
    for (int i = 0; i < b.m_row; i++)
        for (int j = 0; j < b.m_col; j++)
            ASSERT_EQ(b[i][j], b.toArray()[i][j]);
}

TEST(MatrixTest, From_OpenCV_Mat)
{
    // 6 x 2 zeros matrix
    {
        const int aRow = 6, aCol = 2;
        cv::Mat a = cv::Mat::zeros(aRow, aCol, CV_8UC1);
        Matrix<uchar> aMatix = Matrix<uchar>::fromOpenCV(a);
        ASSERT_EQ(aMatix.m_row, aRow);
        ASSERT_EQ(aMatix.m_col, aCol);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatix[i][j], a.at<uchar>(i, j));
                ASSERT_EQ(aMatix[i][j], 0);
            }
        }
    }
    // 4 x 4 ones matrix
    {
        const int aRow = 4, aCol = 4;
        cv::Mat a = cv::Mat::ones(aRow, aCol, CV_8UC1);
        Matrix<uchar> aMatix = Matrix<uchar>::fromOpenCV(a);
        ASSERT_EQ(aMatix.m_row, aRow);
        ASSERT_EQ(aMatix.m_col, aCol);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatix[i][j], a.at<uchar>(i, j));
                ASSERT_EQ(aMatix[i][j], 1);
            }
        }
    }
    // 10 x 10 float random value matrix
    {
        const int aRow = 10, aCol = 10;
        double low = -500.0;
        double high = +500.0;
        cv::Mat a = cv::Mat(aRow, aCol, CV_32F);
        randu(a, cv::Scalar(low), cv::Scalar(high));
        Matrix<float> aMatix = Matrix<float>::fromOpenCV(a);
        ASSERT_EQ(aMatix.m_row, aRow);
        ASSERT_EQ(aMatix.m_col, aCol);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatix[i][j], a.at<float>(i, j));
            }
        }
    }
}

TEST(MatrixTest, To_OpenCV_Mat)
{
    // 5 x 10 int matrix
    {
        const int aRow = 5, aCol = 10;
        Matrix<int> aMatrix(aRow, aCol);
        int cnt = 0;
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                aMatrix[i][j] = cnt++;
            }
        }
        cv::Mat* cvMat = aMatrix.toOpenCVMat(CV_32S);
        Matrix<int> bMatrix = Matrix<int>::fromOpenCV(*cvMat);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatrix[i][j], bMatrix[i][j]);
            }
        }
    }
    // 20 x 10 float random value matrix
    {
        const int aRow = 20, aCol = 10;
        double low = -500.0;
        double high = +500.0;
        cv::Mat a = cv::Mat(aRow, aCol, CV_32F);
        randu(a, cv::Scalar(low), cv::Scalar(high));
        Matrix<float> aMatrix = Matrix<float>::fromOpenCV(a);
        cv::Mat* cvMat = aMatrix.toOpenCVMat(CV_32S);
        Matrix<float> bMatrix = Matrix<float>::fromOpenCV(*cvMat);
        for (int i = 0; i < aRow; i++) {
            for (int j = 0; j < aCol; j++) {
                ASSERT_EQ(aMatrix[i][j], bMatrix[i][j]);
            }
        }
    }
}

TEST(MatrixTest, conv2D)
{
    Matrix<int> inputDim4({
        {5, 7, 1, 2},
        {4, 3, 9, 0},
        {8, 7, 6, 1},
        {4, 2, 0, 7}
    });
    Matrix<int> kernelDim3({
        {1, 0, 1},
        {1, 1, 1},
        {2, 1, 0}
    });
    // 4 x 4 input, 3 x 3 kernel, 1 stride, valid
    {
        
        Matrix<int> *result = Matrix<int>::conv2D(inputDim4, kernelDim3, 1, false);
        Matrix<int> expected({
            {45, 41},
            {44, 21}
        });
        assertMatrixEqual(*result, expected);
    }
    // 4 x 4 input, 3 x 3 kernel, 1 stride, same
    {
        
        Matrix<int> *result = Matrix<int>::conv2D(inputDim4, kernelDim3, 1, true);
        Matrix<int> expected({
            {16, 24, 25, 21},
            {22, 45, 41, 23},
            {22, 44, 21, 23},
            {13, 20, 17, 13}
        });
        assertMatrixEqual(*result, expected);
    }
    // 4 x 4 input, 3 x 3 kernel, 2 stride, valid
    {
        
        Matrix<int> *result = Matrix<int>::conv2D(inputDim4, kernelDim3, 2, false);
        vector<vector<int>> v {
            {45}
        };
        Matrix<int> expected(v);
        assertMatrixEqual(*result, expected);
    }
    // 4 x 4 input, 3 x 3 kernel, 2 stride, same
    {
        
        Matrix<int> *result = Matrix<int>::conv2D(inputDim4, kernelDim3, 2, true);
        vector<vector<int>> v {
            {16, 25},
            {22, 21}
        };
        Matrix<int> expected(v);
        assertMatrixEqual(*result, expected);
    }
}