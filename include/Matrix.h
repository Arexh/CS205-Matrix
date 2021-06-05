#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <math.h>

using namespace std;
using std::setw;

template <class T>
class Matrix : public std::vector<std::vector<T>>
{
public:
    int m_row;
    int m_col;
    Matrix(int row, int col);
    Matrix(const Matrix &a);
    Matrix<T> operator=(Matrix &m1); //深拷贝
    bool operator==(Matrix &m1);     //矩阵相同时true
    bool operator!=(Matrix &m1);     //矩阵不相同时true
    Matrix<T> operator+(Matrix &m1);
    Matrix<T> operator-(Matrix &m1);
    Matrix<T> operator*(Matrix &m1);
    Matrix<T> operator*(int a);
    Matrix<T> operator/(double a);

    Matrix<T> operator+=(Matrix &m1);
    Matrix<T> operator-=(Matrix &m1);
    Matrix<T> operator*=(Matrix &m1);
    Matrix<T> operator*=(int a);
    Matrix<T> operator/=(int a);
    Matrix<T> operator^(Matrix &m1); //矩阵按位置相乘
    Matrix<T> operator~();           //取共轭矩阵

    Matrix<T> dot(Matrix &m1);   //未实现
    Matrix<T> cross(Matrix &m1); //未实现
    Matrix<T> Transposition();
    Matrix<T> toTransposition();

    T determinant();
    T all_sort(int a[], int now, int length, T &determinant);

    T trace();
    Matrix<T> LU_factor_U();
    Matrix<T> LU_factor_L();
    Matrix<T> LDU_factor_L();
    Matrix<T> LDU_factor_D();
    Matrix<T> LDU_factor_U();

    T sum(int row = 0, int col = 0);
    T mean(int row = 0, int col = 0);
    T max(int row = 0, int col = 0);
    T min(int row = 0, int col = 0);

    void printMatrix();
};