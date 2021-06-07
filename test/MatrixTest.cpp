#include "gtest/gtest.h"
#include "Matrix.h"
#include <complex>

using namespace std;

TEST(MatrixTest, MatrixAdd)
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

    Matrix<complex<double>> aa({ { (1.0 + 2i), (3.0 + 4i),(5.0 + 6i), (7.0 + 8i)},
                           { (8.0 + 7i), (6.0 + 5i),(4.0 + 3i), (2.0 + 1i)} });
    Matrix<complex<double>> bb({ { (8.0 + 7i), (6.0 + 5i),(4.0 + 3i), (2.0 + 1i)},
                            { (1.0 + 2i), (3.0 + 4i),(5.0 + 6i), (7.0 + 8i)} });
    Matrix<complex<double>> cc = aa + bb;
    for (int i = 0; i < cc.m_row; i++)
    {
        for (int j = 0; j < cc.m_row; j++)
        {
            ASSERT_EQ(cc[i][j], 9.0 + 9i);
        }
    }
}

TEST(MatrixTest, MatrixMinus)
{
    Matrix<int> a({ {1, 2, 3, 4},
                   {5, 6, 7, 8} });
    Matrix<int> b({ {2, 3, 4, 5},
                   {6, 7, 8, 9} });
    Matrix<int> c = a - b;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            ASSERT_EQ(c[i][j], -1);
        }
    }

    Matrix<complex<double>> aa({ { (1.0 + 2i), (3.0 + 4i)},
                           { (5.0 + 6i), (7.0 + 8i)} });
    Matrix<complex<double>> bb({ { (2.0 + 3i), (4.0 + 5i)},
                            { (6.0 + 7i), (8.0 + 9i)} });
    Matrix<complex<double>> cc = aa - bb;
    for (int i = 0; i < cc.m_row; i++)
    {
        for (int j = 0; j < cc.m_row; j++)
        {
            ASSERT_EQ(-1.0-1i, (aa-bb)[i][j]);
        }
    }
}

TEST(MatrixTest, TestMatrixMultiply)
{
    Matrix<int> a({{ 1,1 },
                    { 2,2 }});
    Matrix<int> b({ { -2,-2 },
                    { 1,1 } });
    Matrix<int> c({ {-1,-1},
                    {-2,-2} });
    for (int i = 0; i < c.m_row; i++)
    {
        for (int j = 0; j < c.m_row; j++)
        {
            ASSERT_EQ(c[i][j], (a*b)[i][j]);
        }
    }

    Matrix<complex<int>> aa({ { complex<int>(1,1),complex<int>(1,1) },
                            { complex<int>(2,2),complex<int>(2,2) } });
    Matrix<complex<int>> bb({ { complex<int>(-2,-2),complex<int>(-2,-2) },
                            { complex<int>(1,1),complex<int>(1,1) } });
    (aa * bb).printMatrix();
    Matrix<complex<int>> mul({ { complex<int>(0,-2),complex<int>(0,-2) },
                            { complex<int>(0,-4),complex<int>(0,-4) } });
    for (int i = 0; i < mul.m_row; i++)
    {
        for (int j = 0; j < mul.m_row; j++)
        {
            ASSERT_EQ(mul[i][j], (aa * bb)[i][j]);
        }
    }
}

TEST(MatrixTest, DefautConstructorTest)
{   //无参数
    Matrix<int> a;
    a.printMatrix();
    Matrix<complex<double>> *b = new Matrix<complex<double>>;
    b->printMatrix();
    delete b;
    //有参数
    Matrix<double> c(3, 3);
    c.printMatrix();
    Matrix<complex<int>> d(5, 5);
    d.printMatrix();
    Matrix<complex<double>>* e = new Matrix<complex<double>>(4,4);
    e->printMatrix();
    delete e;
}

TEST(MatrixTest, CopyConstructorTest)
{   
    int i, j;
    Matrix<int> a({ {1, 2, 3, 4},
                   {5, 6, 7, 8} });
    Matrix<int> aa(a);
    for (i = 0; i < a.m_row; i++)
        for (j = 0; j < a.m_col; j++)
            ASSERT_EQ(aa[i][j], a[i][j]);

    ASSERT_NE(&a, &aa);
    //////////////////
    Matrix<complex<double>> b({ { (1.0 + 2i), (3.0 + 4i),(5.0 + 6i), (7.0 + 8i)},
                           { (8.0 + 7i), (6.0 + 5i),(4.0 + 3i), (2.0 + 1i)} });
    Matrix<complex<double>> bb(b);
    for (i = 0; i < b.m_row; i++)
        for (j = 0; j < b.m_col; j++)
            ASSERT_EQ(bb[i][j], b[i][j]);

    ASSERT_NE(&b, &bb);
    /////////////////
    Matrix<complex<int>> c({ {complex<int>(1,2),complex<int>(3,4)},
                        {complex<int>(5,6),complex<int>(7,8)} });
    Matrix<complex<int>> cc(c);
    for (i = 0; i < c.m_row; i++)
        for (j = 0; j < c.m_col; j++)
            ASSERT_EQ(cc[i][j], c[i][j]);

    ASSERT_NE(&c, &cc);
}

TEST(MatrixTest, row_col_negative)
{
    Matrix<complex<int>> a(-3, -1);
    Matrix<int> b(0, -5);
    Matrix<double> c(-5, 0);
    Matrix<complex<float>> d(1, 0);
    //检测复制构造器
    Matrix<complex<int>> aa(a);
    Matrix<int> bb(b);

    ASSERT_GE(a.m_row, 0)<<"m_row is negetive";
    ASSERT_GE(a.m_col, 0)<<"m_col is negetive";
    ASSERT_GE(b.m_row, 0)<<"m_row is negetive";
    ASSERT_GE(b.m_col, 0)<<"m_col is negetive";
    ASSERT_GE(c.m_row, 0)<<"m_row is negetive";
    ASSERT_GE(c.m_col, 0)<<"m_col is negetive";
    ASSERT_GE(d.m_row, 0)<<"m_row is negetive";
    ASSERT_GE(d.m_col, 0)<<"m_col is negetive";
    ASSERT_GE(aa.m_row, 0) << "m_row is negetive";
    ASSERT_GE(aa.m_col, 0) << "m_col is negetive";
    ASSERT_GE(bb.m_row, 0) << "m_row is negetive";
    ASSERT_GE(bb.m_col, 0) << "m_col is negetive";
}

TEST(MatrixTest, AssignmentOperator)
{
    int i, j;
    Matrix<complex<int>> a({{complex<int>(1,2),complex<int>(3,4)},
                            {complex<int>(5,6),complex<int>(7,8)}});
    Matrix<complex<int>> aa = a;
    for (i = 0; i < a.m_row; i++)
        for (j = 0; j < a.m_col; j++)
            ASSERT_EQ(aa[i][j], a[i][j]);
    ASSERT_NE(&a, &aa);

    Matrix<double> b({ {1,2,3,4},
                        {5,6,7,8}});
    Matrix<double> bb = b;
    for (i = 0; i < b.m_row; i++)
        for (j = 0; j < b.m_col; j++)
            ASSERT_EQ(bb[i][j], b[i][j]);
    ASSERT_NE(&b, &bb);
}

TEST(MatrixTest, ComplexConjugate)//取共轭test
{
    int i, j;
    Matrix<complex<int>> c({ {complex<int>(1,2),complex<int>(3,4)},
                    {complex<int>(5,6),complex<int>(7,8)} });
    Matrix<complex<int>> cc({ {complex<int>(1,-2),complex<int>(3,-4)},
                {complex<int>(5,-6),complex<int>(7,-8)} });
    for (i = 0; i <cc.m_row; i++)
        for (j = 0; j < cc.m_col; j++)
            ASSERT_EQ(conj(cc[i][j]), c[i][j]);
}

TEST(MatrixTest, trace)
{
    Matrix<double> a({  {1,1,1,1},
                        {2,2,2,2},
                        {1,2,3,4},
                        {1,5,6,9} });
    ASSERT_EQ(15, a.trace());

    Matrix<double> b({ {5.5,2} });
    ASSERT_EQ(0, b.trace());

    Matrix<complex<double>> d({ {complex<double>(1,1),complex<double>(1,1) },
                        {complex<double>(1,-1),complex<double>(1,1)} });
    ASSERT_EQ(complex<double>(2, 2), d.trace());

}

TEST(MatrixTest, determinant)
{
    Matrix<double> a({  {1,1,1,1},
                        {2,2,2,2},
                        {1,2,3,4},
                        {1,5,6,9} });
    ASSERT_EQ(0, a.determinant());

    Matrix<double> b({{5.5}});
    ASSERT_EQ(5.5,b.determinant());

    Matrix<double> c({  {1,1},
                        {2,3} });
    ASSERT_EQ(1, c.determinant());

    Matrix<complex<double>> d({ {complex<double>(1,1),complex<double>(1,1) },
                            {complex<double>(1,-1),complex<double>(1,1)} });
    ASSERT_EQ(complex<double>(-2, 2), d.determinant());
}

TEST(MatrixTest, Inverse) 
{
    Matrix<double> b({ {4} });
    (b.Inverse()).printMatrix();
    (b * b.Inverse()).printMatrix();

    Matrix<double> c({  {1,-1},
                        {1,1} });
    (c.Inverse()).printMatrix();
    (c * c.Inverse()).printMatrix();

    Matrix<double> d({  {1,0,0},
                        {0,2,0},
                        {0,0,4} });
    (d.Inverse()).printMatrix();
    (d * d.Inverse()).printMatrix();

    Matrix<complex<double>> e({ {complex<double>(1,1),complex<double>(1,1) },
                        {complex<double>(1,-1),complex<double>(1,1)} });
    e.Inverse().printMatrix();
    (e * e.Inverse()).printMatrix();
}

TEST(MatrixTest, Max_Min_Sum_Mean_Test) 
{
    Matrix<double> a({  {5,2,1,1},
                        {2,3,4,5},
                        {1,2,3,4},
                        {1,5,6,9} });
    double sum = (double)(5 + 2 + 1 + 1 + 2 + 3 + 4 + 5 + 1 + 2 + 3 + 4 + 1 + 5 + 6 + 9);
    double mean = ((5 + 2 + 1 + 1 + 2 + 3 + 4 + 5 + 1 + 2 + 3 + 4 + 1 + 5 + 6 + 9)/16.0);
    ASSERT_EQ(sum, a.sum());
    ASSERT_EQ(9, a.max());
    ASSERT_EQ(1, a.min());
    ASSERT_EQ(mean, a.mean());

    double row_sum = (double)(5 + 2 + 1 + 1);
    double row_mean = (double)(5 + 2 + 1 + 1)/4.0;
    ASSERT_EQ(row_sum, a.row_sum(0));
    ASSERT_EQ(5, a.row_max(0));
    ASSERT_EQ(1, a.row_min(0));
    ASSERT_EQ(row_mean, a.row_mean(0));

    double col_sum = (double)(2+3+2+5);
    double col_mean = (double)(2 + 3 + 2 + 5) / 4.0;
    ASSERT_EQ(col_sum, a.col_sum(1));
    ASSERT_EQ(5, a.col_max(1));
    ASSERT_EQ(2, a.col_min(1));
    ASSERT_EQ(col_mean, a.col_mean(1));
}

TEST(MatrxTest, reshape)
{
    Matrix<double> a({  {5,2,1,1,2},
                        {2,3,4,5,3},
                        {1,2,3,4,4},
                        {1,5,6,9,5} });
    a.reshape(2, 10).printMatrix();
    a.reshape(3, 4).printMatrix();//不满足总数则返回原矩阵
}

TEST(MatrxTest, slice)
{
    Matrix<double> a({ {5,2,1,1,2},
                        {2,3,4,5,3},
                        {1,2,3,4,4},
                        {1,5,6,9,5} });
    a.slice(1, 3, 2, 4).printMatrix();
    a.slice(-1, -1, 5, 5).printMatrix();//越界的归为0或4
    a.slice(2, 2, 1, 4).printMatrix();//打印第二行，1~4列的元素
}


