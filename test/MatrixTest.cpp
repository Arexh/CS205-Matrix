#include "gtest/gtest.h"
#include "Matrix.h"
#include <complex>

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

TEST(MatrixTest, TestMatrixMinus)
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

    Matrix<complex<double>> aa({ { (1.0 + 2i), (3.0 + 4i),(5.0 + 6i), (7.0 + 8i)},
                           { (8.0 + 7i), (6.0 + 5i),(4.0 + 3i), (2.0 + 1i)} });
    Matrix<complex<double>> bb({ { (8.0 + 7i), (10.0 + 9i),(12.0 + 11i), (14.0 + 13i)},
                            { (15.0 + 12i), (13.0 + 10i),(11.0 + 8i), (9.0 + 6i)} });
    Matrix<complex<double>> cc = aa - bb;
    for (int i = 0; i < cc.m_row; i++)
    {
        for (int j = 0; j < cc.m_row; j++)
        {
            ASSERT_EQ(cc[i][j], (aa-bb)[i][j]);
        }
    }
}

TEST(MatrixTest, TestMatrixMultiply)
{
    Matrix<int> a({{ 1,2 },
                    { 3,4 }});
    Matrix<int> b({ { 4,3 },
                    { 2,1 } });
    Matrix<int> c({ {8,5},
                    {20,13} });
    for (int i = 0; i < c.m_row; i++)
    {
        for (int j = 0; j < c.m_row; j++)
        {
            ASSERT_EQ(c[i][j], (a*b)[i][j]);
        }
    }

    Matrix<complex<int>> aa({ { complex<int>(1,2),complex<int>(3,4) },
                            { complex<int>(5,6),complex<int>(7,8) } });
    Matrix<complex<int>> bb({ { complex<int>(9,10),complex<int>(11,12) },
                            { complex<int>(13,14),complex<int>(15,16) } });
    (aa * bb).printMatrix();
    Matrix<complex<int>> cc({ { complex<int>(-28,122),complex<int>(-32,142) },
                            { complex<int>(-36,306),complex<int>(-40,358) } });
    for (int i = 0; i < cc.m_row; i++)
    {
        for (int j = 0; j < cc.m_row; j++)
        {
            ASSERT_EQ(cc[i][j], (aa * bb)[i][j]);
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
}

TEST(MatrixTest, CopyConstructorTest)
{   
    Matrix<int> a({ {1, 2, 3, 4},
                   {5, 6, 7, 8} });
    Matrix<int> aa(a);
    aa.printMatrix();

    Matrix<complex<double>> b({ { (1.0 + 2i), (3.0 + 4i),(5.0 + 6i), (7.0 + 8i)},
                           { (8.0 + 7i), (6.0 + 5i),(4.0 + 3i), (2.0 + 1i)} });
    Matrix<complex<double>> bb(b);
    bb.printMatrix();

    Matrix<complex<int>> c({ {complex<int>(1,2),complex<int>(3,4)},
                        {complex<int>(5,6),complex<int>(7,8)} });
    Matrix<complex<int>> cc(c);
    //判断是否为深拷贝
    ASSERT_NE(&a, &aa);
    ASSERT_NE(&b, &bb);
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

TEST(MatrixTest, assignmentOperatorTest)
{
    Matrix<complex<int>> a({{complex<int>(1,2),complex<int>(3,4)},
                            {complex<int>(5,6),complex<int>(7,8)}});
    a.printMatrix();
    Matrix<complex<int>> aa = a;
    aa.printMatrix();

    Matrix<double> b({ {1,2,3,4},
                        {5,6,7,8}});
    Matrix<double> bb = b;
    b.printMatrix();
    ASSERT_NE(&a, &aa);
    ASSERT_NE(&b, &bb);
}




