#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <math.h>
#include <complex>
#include <typeinfo>
#include <random>

using namespace std;
using std::setw;

template <typename T>
struct Node {
    int row;
    int col;
    T value;
};

template <typename T>
class SparseMatrix
{
private:
    vector<Node<T>> *data;
public:
    int m_row;
    int m_col;

    SparseMatrix() : SparseMatrix(0, 0){};
    SparseMatrix(int row, int col);
    SparseMatrix(int row, int col, const T* data);
    SparseMatrix(const vector<vector<T>>& arr);
    SparseMatrix(const SparseMatrix<T> &a);

    ~SparseMatrix();

    bool operator==(const SparseMatrix<T> &m1) const;
    bool operator!=(const SparseMatrix<T> &m1) const;

    bool is_size_equal(const SparseMatrix<T> &m1) const;
    bool is_square() const;
    bool is_zero() const;

    SparseMatrix<T> operator=(const SparseMatrix<T> &m1);

    vector<T>& operator[](int idx) const;

    template <typename U>
    friend SparseMatrix<U> operator+(const SparseMatrix<U>& l, const SparseMatrix<U>& r);
    template <typename U>
    friend SparseMatrix<U> operator+(const U& l, const SparseMatrix<U>& r);
    template <typename U>
    friend SparseMatrix<U> operator+(const SparseMatrix<U>& l, const U& r);

    template <typename U>
    friend SparseMatrix<U> operator-(const SparseMatrix<U>& l, const SparseMatrix<U>& r);
    template <typename U>
    friend SparseMatrix<U> operator-(const U& l, const SparseMatrix<U>& r);
    template <typename U>
    friend SparseMatrix<U> operator-(const SparseMatrix<U>& l, const U& r);

    template <typename U>
    friend SparseMatrix<U> operator*(const SparseMatrix<U>& l, const SparseMatrix<U>& r);
    template <typename U>
    friend SparseMatrix<U> operator*(const U& l, const SparseMatrix<U>& r);
    template <typename U>
    friend SparseMatrix<U> operator*(const SparseMatrix<U>& l, const U& r);
    
    template <typename U>
    friend SparseMatrix<U> operator/(const U& l, const SparseMatrix<U>& r);
    template <typename U>
    friend SparseMatrix<U> operator/(const SparseMatrix<U>& l, const U& r);
    template <typename U>
    friend SparseMatrix<U> operator^(const SparseMatrix<U>& l, const SparseMatrix<U>& r);

    SparseMatrix<T> operator-() const;

    SparseMatrix<T> operator+=(const SparseMatrix<T> &m1);
    SparseMatrix<T> operator-=(const SparseMatrix<T> &m1);
    SparseMatrix<T> operator*=(const SparseMatrix<T> &m1);
    SparseMatrix<T> operator*=(const T& a);
    SparseMatrix<T> operator/=(const T& a);

    SparseMatrix<T> conju();

    SparseMatrix<T> & dot(const SparseMatrix<T> &m1) const;
    SparseMatrix<T> cross(const SparseMatrix<T> &m1) const;
    SparseMatrix<T> Transposition() const;
    SparseMatrix<T> toTransposition();

    T determinant() const;
    T trace() const;

    SparseMatrix<T> LU_factor_U() const;
    SparseMatrix<T> LU_factor_L() const;
    SparseMatrix<T> LDU_factor_L() const;
    SparseMatrix<T> LDU_factor_D() const;
    SparseMatrix<T> LDU_factor_U() const;

    SparseMatrix<T> Inverse() const;
    SparseMatrix<T> reshape(int r, int c) const;
    SparseMatrix<T> slice(int r1, int r2, int c1, int c2) const;

    T sum() const;
    T mean() const;
    T max() const;
    T min() const;
    T row_max(int row) const;
    T row_min(int row) const;
    T row_sum(int row) const;
    T row_mean(int row) const;

    T col_max(int col) const;
    T col_min(int col) const;
    T col_sum(int col) const;
    T col_mean(int col) const;

    void printMatrix() const;

    pair<SparseMatrix<T>, SparseMatrix<T>> QR_decomposition() const;
    T norm() const;
    SparseMatrix<T> normalized() const;
    void SetIdentity();

    T *eigenvalues(int max_iter = 10e3);
    SparseMatrix<T> eigenvector(T eigenvalue, int max_iter = 10e3);
    SparseMatrix<T> *eigenvectors(int max_itr = 10e3);
    bool isUpperTri();
    static bool isCloseEnough(T a, T b, double threshold = EQ_THRESHOLD);
    pair<T, SparseMatrix<T>> eigenValueAndEigenVector(int max_itr = 10e3);
    T** toArray();
    cv::Mat* toOpenCVMat(int type);

    static SparseMatrix<T> fromOpenCV(const cv::Mat &cvMat);
    static SparseMatrix<T> *conv2D(const SparseMatrix<T> &input, const SparseMatrix<T> &kernel, int stride=1, bool same_padding=true);

private:
    T all_sort(int a[], int now, int length, T &determinant) const;
    void printMatrixInt() const;
};

template <typename T>
SparseMatrix<T>::~SparseMatrix()
{
    if (data)
        delete data;
    data = nullptr;
}

template <typename T>
SparseMatrix<T>::SparseMatrix(int row, int col)
{
    if (row <= 0 || col <= 0)
    {
        cout << "You input negative row/col num" << endl;
        m_row = 0;
        m_col = 0;
        data = nullptr;
    }
    else
    {
        m_row = row;
        m_col = col;
        data = new vector<Node<T>>();
    }
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const vector<vector<T>>& arr)
{
    int row = arr.size();
    int col = arr[0].size();
    m_row = row;
    m_col = col;
    data = new vector<Node<T>>();
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++) {
            if (arr[i][j] != 0) {
                Node<T> node = {i, j, arr[i][j]};
                data->push_back(node);
            }
        }
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T> &a)
{
    m_row = a.m_row;
    m_col = a.m_col;
    data = new vector<Node<T>>(*a.data);
}

// template <typename T>
// void SparseMatrix<T>::printMatrix() const
// {
//     printf("\n---------row:%d,col:%d-----------\n", m_row, m_col);
//     for (int i = 0; i < m_row; i++)
//     {
//         for (int j = 0; j < m_col; j++)
//         {
//             cout << std::setw(7) << (*data)[i][j] << " ";
//         }
//         printf("\n");
//     }
//     printf("---------------------------------\n");
// }

// template <typename T>
// void SparseMatrix<T>::printMatrixInt() const
// {
//     printf("\n---------row:%d,col:%d-----------\n", m_row, m_col);
//     for (int i = 0; i < m_row; i++)
//     {
//         for (int j = 0; j < m_col; j++)
//         {
//             cout << setprecision(2) << std::setw(7) << ((int) (*data)[i][j]) << " ";
//         }
//         printf("\n");
//     }
//     printf("---------------------------------\n");
// }

// template <>
// void SparseMatrix<char>::printMatrix() const
// {
//     printMatrixInt();
// }

// template <>
// void SparseMatrix<uchar>::printMatrix() const
// {
//     printMatrixInt();
// }

// template <typename T>
// vector<T>& SparseMatrix<T>::operator[](int idx) const
// {
//     return (*data)[idx];
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator=(const SparseMatrix<T> &m1)
// {
//     if (this != &m1) {
//         m_row = m1.m_row;
//         m_col = m1.m_col;

//         if (data)
//             delete data;
        
//         data = new vector<vector<T>>(m1.m_row);

//         typename vector<vector<T>>::iterator iter;
//         for (iter = data->begin(); iter < data->end(); iter++)
//             iter->resize(m1.m_col);

//         for (int i = 0; i < m1.m_row; i++)
//             for (int j = 0; j < m1.m_col; j++)
//                 (*data)[i][j] = m1[i][j];
//     }
//     return (*this);
// }

// template <typename T>
// bool SparseMatrix<T>::operator==(const SparseMatrix<T> &m1) const
// {
//     if (m_row != m1.m_row || m_col != m1.m_col)
//         return false;

//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             if (!isCloseEnough((*data)[i][j], m1[i][j]))
//                 return false;

//     return true;
// }

// template <typename T>
// bool SparseMatrix<T>::operator!=(const SparseMatrix<T> &m1) const
// {
//     return !(*this == m1);
// }

// template <typename T>
// bool SparseMatrix<T>::is_size_equal(const SparseMatrix<T> &m1) const
// {
//     return m_row == m1.m_row && m_col == m1.m_col;
// }

// template <typename T>
// bool SparseMatrix<T>::is_square() const
// {
//     return m_row == m_col;
// }

// template <typename T>
// bool SparseMatrix<T>::is_zero() const
// {
//     return determinant() ==  static_cast<T>(0);
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator-() const
// {
//     SparseMatrix<T> result(*this);
    
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             result[i][j] = -result[i][j];
    
//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator+(const SparseMatrix<T>& l, const SparseMatrix<T>& r)
// {
//     SparseMatrix<T> result(l);

//     for (int i = 0; i < l.m_row; i++)
//         for (int j = 0; j < l.m_col; j++)
//             result[i][j] += r[i][j];

//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator+(const SparseMatrix<T>& l, const T& r)
// {
//     SparseMatrix<T> result(l);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] += r;

//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator+(const T& l, const SparseMatrix<T>& r)
// {
//     return r + l;
// }

// template <typename T>
// SparseMatrix<T> operator-(const SparseMatrix<T>& l, const SparseMatrix<T>& r)
// {
//     SparseMatrix<T> result(l);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] -= r[i][j];

//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator-(const SparseMatrix<T>& l, const T& r)
// {
//     SparseMatrix<T> result(l);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] -= r;

//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator-(const T& l, const SparseMatrix<T>& r)
// {
//     return r.SparseMatrix<T>::operator-() + l;
// }

// template <typename T>
// SparseMatrix<T> operator*(const SparseMatrix<T>& l, const SparseMatrix<T>& r)
// {
//     SparseMatrix<T> result(l.m_row, r.m_col);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             for (int k = 0; k < l.m_col; k++)
//                 result[i][j] += l[i][k] * r[k][j];

//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator*(const SparseMatrix<T>& l, const T& r)
// {
//     SparseMatrix<T> result(l);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] *= r;

//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator*(const T& l, const SparseMatrix<T>& r)
// {
//     return r * l;
// }

// template <typename T>
// SparseMatrix<T> operator/(const SparseMatrix<T>& l, const T& r)
// {
//     SparseMatrix<T> result(l);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] /= r;

//     return result;
// }

// template <typename T>
// SparseMatrix<T> operator/(const T& l, const SparseMatrix<T>& r)
// {
//     SparseMatrix<T> result(r);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] = l / result[i][j];

//     return result;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator+=(const SparseMatrix<T>& m1)
// {
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             (*data)[i][j] += m1[i][j];

//     return (*this);
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator-=(const SparseMatrix<T> &m1)
// {
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             (*data)[i][j] -= m1[i][j];

//     return (*this);
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator*=(const SparseMatrix<T> &m1)
// {
//     SparseMatrix<T> result = (*this) * m1;
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             (*data)[i][j] = result[i][j];

//     return (*this);
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator*=(const T& m1)
// {
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             (*data)[i][j] *= m1;

//     return (*this);
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator/=(const T& a)
// {
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             (*data)[i][j] = (*data)[i][j] / a;

//     return (*this);
// }

// template <typename T>
// SparseMatrix<T> operator^(const SparseMatrix<T> &l, const SparseMatrix<T> &r)
// {
//     SparseMatrix<T> result(l);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] *= r[i][j];

//     return result;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::conju()
// {
//     SparseMatrix<T> result(*this);

//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             result[i][j] = conj(result[i][j]);

//     return result;
// }

// template <typename T>
// SparseMatrix<T> & SparseMatrix<T>::dot(const SparseMatrix<T> &m1) const
// {
//     return (*this) ^ m1;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::cross(const SparseMatrix<T> &m1) const
// {
//     return (*this) * m1;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::Transposition() const
// {
//     SparseMatrix<T> result(m_col, m_row);

//     for (int i = 0; i < result.m_row; i++)
//         for (int j = 0; j < result.m_col; j++)
//             result[i][j] = (*data)[j][i];

//     return result;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::toTransposition()
// {
//     // SparseMatrix<T> result = Transposition();
    
//     // for (int i = 0; i < m_row; i++)
//     //     for (int j = 0; j < m_col; j++)
//     //         (*data)[i][j] = result[i][j];

//     return (*this);
// }

// template <typename T>
// T SparseMatrix<T>::determinant() const
// {
//     assert(m_col == m_row);
//     int length = m_col, now = 0;
//     T d;
//     d = 0;
//     int *permutation = new int[length];
//     for (int i = 0; i < length; i++)
//     {
//         permutation[i] = i;
//     }

//     d = all_sort(permutation, now, length, d);
//     delete[] permutation;

//     return d;
// }

// template <typename T>
// T SparseMatrix<T>::all_sort(int a[], int now, int length, T &determinant) const
// {
//     if (now == length - 1)
//     {
//         T tmp;
//         tmp = 1;
//         int pow = 0;
//         for (int i = 0; i < length; i++)
//         {
//             tmp *= (*data)[i][a[i]];
//         }

//         for (int i = 1; i < length; i++)
//             for (int j = 0; j < i; j++)
//                 if (a[j] > a[i])
//                     pow++;

//         if (pow % 2 == 0)
//             determinant += tmp;
//         else
//             determinant -= tmp;
//         //将-1的幂化为0,tmp初始化
//         pow = 0;
//         tmp = 1;
//     }
//     for (int i = now; i < length; i++)
//     {
//         int tmp = a[now];
//         a[now] = a[i];
//         a[i] = tmp;
//         all_sort(a, now + 1, length, determinant);
//         tmp = a[now];
//         a[now] = a[i];
//         a[i] = tmp;
//     }
//     return determinant;
// }

// template <typename T>
// T SparseMatrix<T>::trace() const
// {
//     if (is_square())
//     {
//         T trace = static_cast<T>(0);

//         for (int i = 0; i < m_col; i++)
//             trace += (*data)[i][i];

//         return trace;
//     }
//     else
//     {
//         return static_cast<T>(0);
//     }
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::LU_factor_U() const
// {
//     assert(m_col == m_row);
//     int n = m_col;
//     T sum;
//     sum = 0;
//     SparseMatrix<T> l(n, n);
//     SparseMatrix<T> u(n, n);

//     for (int i = 0; i < n; i++) //初始化矩阵L和矩阵U
//         for (int j = 0; j < n; j++)
//         {
//             u[i][j] = 0;
//             if (i == j)
//                 l[i][j] = 1;
//         }

//     for (int i = 0; i < n; i++)
//     {
//         T sum;
//         sum = 0;
//         for (int j = i; j < n; j++)
//         {
//             for (int k = 0; k <= i - 1; k++)
//                 sum += l[i][k] * u[k][j];
//             u[i][j] = (*data)[i][j] - sum; //计算矩阵U
//             sum = 0;
//         }

//         for (int x = i + 1; x < n; x++)
//         {
//             for (int k = 0; k <= i - 1; k++)
//                 sum += l[x][k] * u[k][i];
//             l[x][i] = ((*data)[x][i] - sum) / u[i][i]; //计算矩阵L
//             sum = 0;
//         }
//     }
//     return u;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::LU_factor_L() const
// {
//     assert(m_col == m_row);
//     //需要判断行列式是否为0
//     int n = m_col;
//     T sum;
//     sum = 0;
//     SparseMatrix<T> l(n, n);
//     SparseMatrix<T> u(n, n);

//     for (int i = 0; i < n; i++) //初始化矩阵L和矩阵U
//         for (int j = 0; j < n; j++)
//         {
//             u[i][j] = 0;
//             if (i == j)
//                 l[i][j] = 1;
//         }

//     for (int i = 0; i < n; i++)
//     {
//         T sum;
//         sum = 0;
//         for (int j = i; j < n; j++)
//         {
//             for (int k = 0; k <= i - 1; k++)
//                 sum += l[i][k] * u[k][j];
//             u[i][j] = (*data)[i][j] - sum; //计算矩阵U
//             sum = 0;
//         }

//         for (int x = i + 1; x < n; x++)
//         {
//             for (int k = 0; k <= i - 1; k++)
//                 sum += l[x][k] * u[k][i];
//             l[x][i] = ((*data)[x][i] - sum) / u[i][i]; //计算矩阵L
//             sum = 0;
//         }
//     }
//     return l;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::LDU_factor_L() const
// {
//     SparseMatrix<T> l(this->LU_factor_L());
//     return l;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::LDU_factor_D() const
// {
//     assert(this->m_row == this->m_col);
//     SparseMatrix<T> tmp(this->LU_factor_U());
//     SparseMatrix<T> d(this->m_row, this->m_col);
//     for (int i = 0; i < m_row; i++)
//     {
//         d[i][i] = tmp[i][i];
//     }
//     return d;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::LDU_factor_U() const
// {
//     assert(this->m_row == this->m_col);
//     SparseMatrix<T> u(this->LU_factor_U());
//     for (int i = 0; i < m_row; i++)
//     {
//         for (int j = i; j < m_col; j++)
//         {
//             u[i][j] /= this->LU_factor_U()[i][i];
//         }
//     }
//     return u;
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::Inverse() const
// {
//     T deter = this->determinant();
//     assert(this->is_square());
//     T tmp1;
//     tmp1 = 0;
//     assert(deter != tmp1);
//     if (this->m_row == 1)
//     {
//         vector<vector<T>> v{{static_cast<T>(1) / (*data)[0][0]}};
//         SparseMatrix<T> tmp(v);
//         return tmp;
//     }
//     else
//     {
//         int i, j, k, m, tt = this->m_row, n = tt - 1;
//         SparseMatrix<T> inverse(tt, tt);
//         SparseMatrix<T> tmp(n, n);
//         for (i = 0; i < tt; i++)
//         {

//             for (j = 0; j < tt; j++)
//             {
//                 for (k = 0; k < n; k++)
//                     for (m = 0; m < n; m++)
//                         tmp[k][m] = (*data)[k >= i ? k + 1 : k][m >= j ? m + 1 : m];

//                 T a = tmp.determinant();
//                 if ((i + j) % 2 == 1)
//                 {
//                     a = -a;
//                 };
//                 T b = a / (this->determinant());
//                 inverse[j][i] = b;
//             }
//         }
//         return inverse;
//     }
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::reshape(int r, int c) const
// {
//     if (this->m_row * this->m_col != r * c)
//     {
//         //cout << "ReshapeError:Not The Same Szie" << __FILE__ << __LINE__ << end;
//         return (*this);
//     }
//     else
//     {
//         SparseMatrix<T> ans(r, c);
//         int i, j, x = 0, y = 0;
//         for (i = 0; i < this->m_row; i++)
//         {
//             for (j = 0; j < this->m_col; j++)
//             {
//                 ans[x][y] = (*data)[i][j];
//                 y++;
//                 if (y == c)
//                 {
//                     x++;
//                     y = 0;
//                 }
//             }
//         }
//         return ans;
//     }
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::slice(int r1, int r2, int c1, int c2) const
// {
//     // assert(r1)
//     if (r1 > r2)
//     {
//         int tmp = r1;
//         r1 = r2;
//         r2 = tmp;
//     }
//     if (c1 > c2)
//     {
//         int tmp = c1;
//         c1 = c2;
//         c2 = tmp;
//     }
//     if (r1 < 0)
//     {
//         if (r2 < 0)
//         {
//             r2 = 0;
//         }
//         r1 = 0;
//     }
//     if (r2 >= this->m_row)
//     {
//         if (r1 >= this->m_row)
//         {
//             r1 = this->m_row - 1;
//         }
//         r2 = this->m_row - 1;
//     }
//     if (c1 < 0)
//     {
//         if (c2 < 0)
//         {
//             c2 = 0;
//         }
//         c1 = 0;
//     }
//     if (c2 >= this->m_col)
//     {
//         if (c1 >= this->m_col)
//         {
//             c1 = this->m_col - 1;
//         }
//         c2 = this->m_col - 1;
//     }

//     SparseMatrix<T> tmp(r2 - r1 + 1, c2 - c1 + 1);
//     for (int i = r1; i <= r2; i++)
//     {
//         for (int j = c1; j <= c2; j++)
//         {
//             tmp[i - r1][j - c1] = (*data)[i][j];
//         }
//     }
//     return tmp;
// }

// template <typename T>
// T SparseMatrix<T>::sum() const
// {
//     T sum;
//     sum = 0;
//     for (int i = 0; i < m_row; i++)
//     {
//         for (int j = 0; j < m_col; j++)
//         {
//             sum += (*data)[i][j];
//         }
//     }
//     return (sum);
// }

// template <typename T>
// T SparseMatrix<T>::mean() const
// {
//     T total;
//     total = this->m_row * this->m_col;
//     return (this->sum() / total);
// }

// template <typename T>
// T SparseMatrix<T>::max() const
// {
//     int k = 0, m = 0, i, j;
//     for (i = 0; i < this->m_row; i++)
//         for (j = 0; j < this->m_col; j++)
//             if ((*data)[i][j] > (*data)[k][m])
//             {
//                 k = i;
//                 m = j;
//             }

//     return (*data)[k][m];
// }

// template <typename T>
// T SparseMatrix<T>::min() const
// {
//     int k = 0, m = 0, i = 0, j = 0;
//     for (i = 0; i < this->m_row; i++)
//     {
//         for (j = 0; j < this->m_col; j++)
//         {
//             if ((*data)[k][m] > (*data)[i][j])
//             {
//                 k = i;
//                 m = j;
//             }
//         }
//     }
//     return (*data)[k][m];
// }

// template <typename T>
// T SparseMatrix<T>::row_max(int row) const
// {
//     assert(row >= 0 && row < this->m_row);
//     int k = 0;
//     for (int i = 0; i < this->m_col; i++)
//     {
//         if ((*data)[row][k] < (*data)[row][i])
//             k = i;
//     }
//     return (*data)[row][k];
// }

// template <typename T>
// T SparseMatrix<T>::row_min(int row) const
// {
//     assert(row >= 0 && row < this->m_row);
//     int k = 0;
//     for (int i = 0; i < this->m_col; i++)
//         if ((*data)[row][k] > (*data)[row][i])
//             k = i;

//     return (*data)[row][k];
// }

// template <typename T>
// T SparseMatrix<T>::row_sum(int row) const
// {
//     assert(row >= 0 && row < this->m_row);
//     T row_sum;
//     row_sum = 0;
//     for (int i = 0; i < this->m_col; i++)
//     {
//         row_sum += (*data)[row][i];
//     }
//     return row_sum;
// }

// template <typename T>
// T SparseMatrix<T>::row_mean(int row) const
// {
//     assert(row >= 0 && row < this->m_row);
//     T total;
//     total = (this->m_col);
//     return this->row_sum(row) / total;
// }

// template <typename T>
// T SparseMatrix<T>::col_max(int col) const
// {
//     assert(col >= 0 && col < this->m_col);
//     int k = 0;
//     for (int i = 0; i < this->m_row; i++)
//         if ((*data)[k][col] < (*data)[i][col])
//             k = i;

//     return (*data)[k][col];
// }

// template <typename T>
// T SparseMatrix<T>::col_min(int col) const
// {
//     assert(col >= 0 && col < this->m_col);
//     int k = 0;
//     for (int i = 0; i < this->m_row; i++)
//         if ((*data)[k][col] > (*data)[i][col])
//             k = i;

//     return (*data)[k][col];
// }

// template <typename T>
// T SparseMatrix<T>::col_sum(int col) const
// {
//     assert(col >= 0 && col < this->m_col);
//     T col_sum;
//     col_sum = 0;
//     for (int i = 0; i < this->m_row; i++)
//     {
//         col_sum += (*data)[i][col];
//     }
//     return col_sum;
// }

// template <typename T>
// T SparseMatrix<T>::col_mean(int col) const
// {
//     assert(col >= 0 && col < this->m_col);
//     T total;
//     total = this->m_row;
//     return this->col_sum(col) / total;
// }

// // from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbQR.h
// template <typename T>
// pair<SparseMatrix<T>, SparseMatrix<T>> SparseMatrix<T>::QR_decomposition() const
// {
//     SparseMatrix<T> input(*this);
//     vector<SparseMatrix<T>> plist;
//     for (int j = 0; j < m_row - 1; j++)
//     {
//         SparseMatrix<T> a1(1, m_row - j);
//         SparseMatrix<T> b1(1, m_row - j);

//         for (int i = j; i < m_row; i++)
//         {
//             a1[0][i - j] = input[i][j];
//             b1[0][i - j] = static_cast<T>(0.0);
//         }
//         b1[0][0] = static_cast<T>(1.0);

//         T a1norm = a1.norm();

//         T sgn = -1;
//         if (a1[0][0] < static_cast<T>(0.0))
//         {
//             sgn = 1;
//         }

//         SparseMatrix<T> temp = b1 * sgn * a1norm;
//         SparseMatrix<T> u = a1 - temp;
//         SparseMatrix<T> n = u.normalized();
//         SparseMatrix<T> nTrans = n.Transposition();
//         SparseMatrix<T> I(m_row - j, m_row - j);
//         I.SetIdentity();

//         SparseMatrix<T> temp1 = n * static_cast<T>(2.0);
//         SparseMatrix<T> temp2 = nTrans * temp1;
//         SparseMatrix<T> Ptemp = I - temp2;

//         SparseMatrix<T> P(m_row, m_col);
//         P.SetIdentity();

//         for (int x = j; x < m_row; x++)
//         {
//             for (int y = j; y < m_col; y++)
//             {
//                 P[x][y] = Ptemp[x - j][y - j];
//             }
//         }

//         plist.push_back(P);
//         input = P * input;
//     }

//     SparseMatrix<T> qMat = plist[0];
//     for (int i = 1; i < m_row - 1; i++)
//     {
//         SparseMatrix<T> temp3 = plist[i].Transposition();
//         qMat = qMat * temp3;
//     }

//     int numElements = plist.size();
//     SparseMatrix<T> rMat = plist[numElements - 1];
//     for (int i = (numElements - 2); i >= 0; i--)
//     {
//         rMat = rMat * plist[i];
//     }
//     rMat = rMat * (*this);

//     return pair<SparseMatrix<T>, SparseMatrix<T>>(qMat, rMat);
// }

// template <typename T>
// T SparseMatrix<T>::norm() const
// {
//     T cumulativeSum = static_cast<T>(0.0);
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             cumulativeSum += (*this)[i][j] * (*this)[i][j];

//     return sqrt(cumulativeSum);
// }

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::normalized() const
// {
//     T norm = this->norm();
//     SparseMatrix<T> copy(*this);
//     return copy * (static_cast<T>(1.0) / norm);
// }

// template <typename T>
// void SparseMatrix<T>::SetIdentity()
// {
//     for (int i = 0; i < m_row; i++)
//         for (int j = 0; j < m_col; j++)
//             if (i == j)
//                 (*this)[i][j] = static_cast<T>(1.0);
//             else
//                 (*this)[i][j] = static_cast<T>(0.0);
// }

// // from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
// // only work for symmetric metrices
// template <typename T>
// T *SparseMatrix<T>::eigenvalues(int max_iter)
// {
//     SparseMatrix<T> A = (*this);
//     SparseMatrix<T> identitySparseMatrix(m_row, m_col);
//     identitySparseMatrix.SetIdentity();

//     for (int i = 0; i < max_iter; i++)
//     {
//         auto qrResult = A.QR_decomposition();
//         A = qrResult.second * qrResult.first;
//         if (A.isUpperTri())
//             break;
//     }

//     T *eigenvalues = new T[m_row];
//     for (int i = 0; i < m_row; i++)
//     {
//         eigenvalues[i] = A[i][i];
//     }
//     return eigenvalues;
// }

// template <typename T>
// bool SparseMatrix<T>::isCloseEnough(T a, T b, double threshold)
// {
//     return abs(a - b) < static_cast<T>(threshold);
// }

// template <>
// bool SparseMatrix<uchar>::isCloseEnough(uchar a, uchar b, double threshold)
// {
//     return a == b;
// }

// template <>
// bool SparseMatrix<char>::isCloseEnough(char a, char b, double threshold)
// {
//     return a == b;
// }


// template <typename T>
// bool SparseMatrix<T>::isUpperTri()
// {
//     T cumulativeSum = static_cast<T>(0);
//     for (int i = 1; i < m_row; i++)
//     {
//         for (int j = 0; j < i; j++)
//         {
//             cumulativeSum += (*this)[i][j];
//         }
//     }
//     return isCloseEnough(cumulativeSum, static_cast<T>(0));
// }

// // from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::eigenvector(T eigenvalue, int max_iter)
// {
//     SparseMatrix<T> A = (*this);
//     random_device myRandomDevice;
//     mt19937 myRandomGenerator(myRandomDevice());
//     uniform_int_distribution<int> myDistribution(1.0, 10.0);

//     SparseMatrix<T> identitySparseMatrix(m_row, m_col);
//     identitySparseMatrix.SetIdentity();

//     SparseMatrix<T> v(m_row, 1);
//     for (int i = 0; i < m_row; i++)
//     {
//         v[i][0] = static_cast<T>(myDistribution(myRandomGenerator));
//     }

//     T deltaThreshold = static_cast<T>(EQ_THRESHOLD);
//     T delta = static_cast<T>(1e-6);
//     SparseMatrix<T> preVector(m_row, 1);
//     SparseMatrix<T> tempSparseMatrix(m_row, m_row);

//     for (int i = 0; i < max_iter; i++)
//     {
//         preVector = v;
//         SparseMatrix<T> temp = identitySparseMatrix * eigenvalue;
//         tempSparseMatrix = A - temp;
//         tempSparseMatrix = tempSparseMatrix.Inverse();
//         v = tempSparseMatrix * v;
//         v = v.normalized();

//         delta = (v - preVector).norm();
//         if (delta > deltaThreshold)
//             break;
//     }
//     return v;
// }

// template <typename T>
// SparseMatrix<T> *SparseMatrix<T>::eigenvectors(int max_itr)
// {
//     SparseMatrix<T> *eigenvectors = new SparseMatrix<T>[m_row];
//     T *eigenvalues = this->eigenvalues();
//     for (int i = 0; i < m_row; i++)
//     {
//         eigenvectors[i] = this->eigenvector(*(eigenvalues + i));
//     }
//     return eigenvectors;
// }

// // from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
// template <typename T>
// pair<T, SparseMatrix<T>> SparseMatrix<T>::eigenValueAndEigenVector(int max_itr)
// {
//     T eigenvalue;
//     SparseMatrix<T> inputSparseMatrix = (*this);
//     random_device myRandomDevice;
//     mt19937 myRandomGenerator(myRandomDevice());
//     uniform_int_distribution<int> myDistribution(1.0, 10.0);
//     SparseMatrix<T> identitySparseMatrix(m_row, m_col);
//     identitySparseMatrix.SetIdentity();

//     SparseMatrix<T> v(m_row, 1);
//     for (int i = 0; i < m_row; i++)
//     {
//         v[i][0] = static_cast<T>(static_cast<T>(myDistribution(myRandomGenerator)));
//     }
//     SparseMatrix<T> v1(m_row, 1);
//     for (int i = 0; i < max_itr; i++)
//     {
//         v1 = inputSparseMatrix * v;
//         v1 = v1.normalized();
//         v = v1;
//     }
//     T cumulativeSum = static_cast<T>(0.0);
//     for (int i = 1; i < m_row; i++)
//     {
//         cumulativeSum += inputSparseMatrix[0][i] * v1[i][0];
//     }
//     eigenvalue = (cumulativeSum / v1[0][0]) + inputSparseMatrix[0][0];
//     return pair<T, SparseMatrix<T>>(eigenvalue, v1);
// }

// // learn from: https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::fromOpenCV(const cv::Mat &cvMat)
// {
//     int row = cvMat.rows;
//     int col = cvMat.cols;
//     SparseMatrix<T> result(row, col);
//     cv::MatConstIterator_<T> it = cvMat.begin<T>(), it_end = cvMat.end<T>();
//     vector<T> array;
//     if (cvMat.isContinuous())
//     {
//         array.assign((T *)cvMat.data, (T *)cvMat.data + cvMat.total() * cvMat.channels());
//     }
//     else
//     {
//         for (int i = 0; i < cvMat.rows; ++i)
//         {
//             array.insert(array.end(), cvMat.ptr<T>(i), cvMat.ptr<T>(i) + cvMat.cols * cvMat.channels());
//         }
//     }
//     int cnt = 0;
//     for (int i = 0; i < row; i++)
//     {
//         for (int j = 0; j < col; j++)
//         {
//             result[i][j] = array[cnt++];
//         }
//     }
//     return result;
// }

// template <typename T>
// T** SparseMatrix<T>::toArray() {
//     T** array = new T*[m_row];
//     for (int i = 0; i < m_row; i++) {
//         array[i] = new T[m_col];
//     }
//     for (int i = 0; i < m_row; i++) {
//         for (int j = 0; j < m_col; j++) {
//             array[i][j] = (*this)[i][j];
//         }
//     }
//     return array;
// }

// template <typename T>
// cv::Mat* SparseMatrix<T>::toOpenCVMat(int type) {
//     cv::Mat* cvMat = new cv::Mat(m_row, m_col, type);
//     for (int i = 0; i < m_row; i++) {
//         for (int j = 0; j < m_col; j++) {
//             (*cvMat).at<T>(i, j) = (*this)[i][j];
//         }
//     }
//     return cvMat;
// }

// template <typename T>
// SparseMatrix<T>* SparseMatrix<T>::conv2D(const SparseMatrix<T> &input, const SparseMatrix<T> &kernel, int stride, bool same_padding) {
//     SparseMatrix<T> inputSparseMatrix;
//     int padding = 0;
//     if (same_padding) {
//         padding = 1;
//     }
//     inputSparseMatrix = SparseMatrix<T>(input.m_row + padding * 2, input.m_col + padding * 2);
//     for (int i = 0; i < input.m_row; i++) {
//         for (int j = 0; j < input.m_col; j++) {
//             inputSparseMatrix[i + padding][j + padding] = input[i][j];
//         }
//     }
//     if (padding == 1) {
//         for (int i = 0; i < inputSparseMatrix.m_row; i++) inputSparseMatrix[i][0] = inputSparseMatrix[i][inputSparseMatrix.m_col - 1] = 0;
//         for (int i = 0; i < inputSparseMatrix.m_col; i++) inputSparseMatrix[0][i] = inputSparseMatrix[inputSparseMatrix.m_row - 1][i] = 0;
//     }
//     int rowDim = ((input.m_row + 2 * padding - kernel.m_row) / stride) + 1;
//     int colDim = ((input.m_col + 2 * padding - kernel.m_col) / stride) + 1;
//     SparseMatrix<T> *result = new SparseMatrix<T>(rowDim, colDim);
//     for (int i = 0; i < rowDim; i++) {
//         for (int j = 0; j < colDim; j++) {
//             T cumulativeSum = static_cast<T>(0);
//             for (int x = 0; x < kernel.m_row; x++) {
//                 for (int y = 0; y < kernel.m_col; y++) {
//                     cumulativeSum += kernel[x][y] * inputSparseMatrix[x + i * stride][y + j * stride];
//                 }
//             }
//             (*result)[i][j] = cumulativeSum;
//         }
//     }
//     return result;
// }