// Artem Matevosian
// a.matevosian@innopolis.university
#include <vector>
#include <ostream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>

#include "gnuplot-iostream.h"

using namespace std;

class Matrix;
class SquareMatrix;
class IdentityMatrix;
class EliminationMatrix;
class PermutationMatrix;


class Matrix {
protected:
    vector<vector<double>> table;
    int width;
    int height;
    vector<vector<double>> transposedTable;

public:
    Matrix(int height, int width) {
        this->height = height;
        this->width = width;
        // ensuring n by m capacity
        table.resize(height);
        for (int i = 0; i < height; i++) {
            table[i].resize(width);
        }
        // ensuring m by n capacity for transposed table
        transposedTable.resize(width);
        for (int i = 0; i < width; i++) {
            transposedTable[i].resize(height);
        }
    }

    int getWidth() {
        return width;
    }

    int getHeight() {
        return height;
    }

    vector<vector<double>> getTable() {
        return table;
    }

    vector<vector<double>> getTransposedTable() {
        return transposedTable;
    }

    void setTables(vector<vector<double>> newTable, vector<vector<double>> newTransposedTable) {
        this->table = newTable;
        this->transposedTable = newTransposedTable;
    }

    void setElement(int rowNumber, int columnNumber, double element) {
        table[rowNumber][columnNumber] = element;
        transposedTable[columnNumber][rowNumber] = element;
    }

    double getElement(int rowNumber, int columnNumber) {
        return table[rowNumber][columnNumber];
    }

    vector<double> getRow(int rowNumber) {
        return table[rowNumber];
    }

    vector<double> getColumn(int columnNumber) {
        return transposedTable[columnNumber];
    }

    bool operator==(Matrix otherMatrix) {
        return width == otherMatrix.getWidth() &&
               height == otherMatrix.getHeight() &&
               table == otherMatrix.getTable();
    }

    bool operator!=(Matrix otherMatrix) {
        return !(otherMatrix == *this);
    }

    Matrix& operator=(Matrix otherMatrix) {
        this->width = otherMatrix.getWidth();
        this->height = otherMatrix.getHeight();
        this->table = otherMatrix.getTable();
        this->transposedTable = otherMatrix.transposedTable;
        return *this;
    }

    Matrix operator+(Matrix otherMatrix) {
        if (this->width != otherMatrix.getWidth() || this->height != otherMatrix.getHeight()) {
            cout << "Error: the dimensional problem occurred" << endl;
            return Matrix(0, 0);
        } else {
            Matrix result = Matrix(this->height, this->width);
            for (int rowNumber = 0; rowNumber < this->height; rowNumber++) {
                for (int columnNumber = 0; columnNumber < this->width; columnNumber++) {
                    double element = this->getElement(rowNumber, columnNumber) +
                                  otherMatrix.getElement(rowNumber, columnNumber);
                    result.setElement(rowNumber, columnNumber, element);
                }
            }
            return result;
        }
    }

    Matrix operator-(Matrix otherMatrix) {
        if (this->width != otherMatrix.getWidth() || this->height != otherMatrix.getHeight()) {
            cout << "Error: the dimensional problem occurred" << endl;
            return Matrix(0, 0);
        } else {
            Matrix result = Matrix(this->height, this->width);
            for (int rowNumber = 0; rowNumber < this->height; rowNumber++) {
                for (int columnNumber = 0; columnNumber < this->width; columnNumber++) {
                    double element = this->getElement(rowNumber, columnNumber) -
                                  otherMatrix.getElement(rowNumber, columnNumber);
                    result.setElement(rowNumber, columnNumber, element);
                }
            }
            return result;
        }
    }

    virtual Matrix operator*(Matrix otherMatrix) {
        if (this->width != otherMatrix.height) {
            cout << "Error: the dimensional problem occurred" << endl;
            return Matrix(0, 0);
        } else {
            Matrix result = Matrix(this->height, otherMatrix.getWidth());
            for (int rowNumber = 0; rowNumber < this->height; rowNumber++) {
                for (int columnNumber = 0; columnNumber < otherMatrix.getWidth(); columnNumber++) {
                    double element = 0;
                    vector<double> row = this->getRow(rowNumber);
                    vector<double> column = otherMatrix.getColumn(columnNumber);
                    for (int i = 0; i < this->width; i++) {
                        element += row[i] * column[i];
                    }
                    result.setElement(rowNumber, columnNumber, element);
                }
            }
            return result;
        }
    }

    Matrix transpose() {
        Matrix result = Matrix(this->width, this->height);
        result.setTables(this->transposedTable, this->table);
        return result;
    }

    // overloading this operator enables output of matrices via cout and file output streams
    friend ostream &operator<<(ostream &os, Matrix matrix) {
        for (int rowNumber = 0; rowNumber < matrix.height; rowNumber++) {
            for (int columnNumber = 0; columnNumber < matrix.width - 1; columnNumber++) {
                double outputElem = matrix.getElement(rowNumber, columnNumber);
                if (abs(outputElem) <= 1e-10) outputElem = 0;
                os << outputElem << " ";
            }
            double outputElem = matrix.getElement(rowNumber, matrix.width - 1);
            if (abs(outputElem) <= 1e-10) outputElem = 0;
            os << outputElem << endl;
        }
        return os;
    }

    // overloading this operator enables input of matrices via cin and file input streams
    friend istream &operator>>(istream &is, Matrix &matrix) {
        for (int rowNumber = 0; rowNumber < matrix.height; rowNumber++) {
            for (int columnNumber = 0; columnNumber < matrix.width; columnNumber++) {
                double element;
                is >> element;
                matrix.setElement(rowNumber, columnNumber, element);
            }
        }
        return is;
    }
};

class SquareMatrix : public Matrix {
public:
    explicit SquareMatrix(int side) : Matrix(side, side) {}

    explicit SquareMatrix(Matrix matrix) : Matrix(matrix.getHeight(), matrix.getWidth()) {
        if (matrix.getWidth() != matrix.getHeight()) {
            cout << "Error: the dimensional problem occurred" << endl;
        } else {
            this->table = matrix.getTable();
            this->transposedTable = matrix.getTransposedTable();
        }
    }

    SquareMatrix operator+(SquareMatrix otherMatrix) {
        Matrix* thisMatrix = (Matrix*)this;
        Matrix result = thisMatrix->operator+((Matrix)otherMatrix);
        return SquareMatrix(result);
    }

    SquareMatrix operator-(SquareMatrix otherMatrix) {
        Matrix* thisMatrix = (Matrix*)this;
        Matrix result = thisMatrix->operator-((Matrix)otherMatrix);
        return SquareMatrix(result);
    }

    SquareMatrix operator*(SquareMatrix otherMatrix) {
        Matrix* thisMatrix = (Matrix*)this;
        Matrix result = thisMatrix->operator*((Matrix)otherMatrix);
        return SquareMatrix(result);
    }

    Matrix operator*(Matrix otherMatrix) override {
        Matrix* thisMatrix = (Matrix*)this;
        Matrix result = thisMatrix->Matrix::operator*(otherMatrix);
        return result;
    }
};

class IdentityMatrix : public SquareMatrix {
public:
    explicit IdentityMatrix(int side) : SquareMatrix(side) {
        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++) {
                if (i == j) setElement(i, j, 1);
                else setElement(i, j, 0);
            }
        }
    }
};

class EliminationMatrix : public IdentityMatrix {
public:
    // elimI and elimJ mean that element with (elimI, elimJ) indices will be nullified after multiplication
    EliminationMatrix(Matrix eliminated, int elimI, int elimJ) : IdentityMatrix(eliminated.getHeight()) {
        double eliminator = -eliminated.getElement(elimI, elimJ) / eliminated.getElement(elimJ, elimJ);
        setElement(elimI, elimJ, eliminator);
    }
};

class PermutationMatrix : public IdentityMatrix {
public:
    // i-th and j-th row will be permuted after multiplication with this matrix
    PermutationMatrix(int side, int i, int j) : IdentityMatrix(side) {
        setElement(i, i, 0);
        setElement(j, j, 0);
        setElement(i, j, 1);
        setElement(j, i, 1);
    }
};

class ColumnVector : public Matrix {
public:
    explicit ColumnVector(int length) : Matrix(length, 1) {}

    // Decorator pattern. Transforms Martix(n x 1) into a column vector
    explicit ColumnVector(Matrix matrix) : Matrix(matrix.getHeight(), 1) {
        if (matrix.getWidth() != 1) {
            cout << "Error: the dimensional problem occurred" << endl;
        } else {
            this->table = matrix.getTable();
            this->transposedTable = matrix.getTransposedTable();
        }
    }

    double dot(ColumnVector vector) {
        double result = 0;
        for (int i = 0; i < height; i++) {
            result += this->table[i][0] * vector.table[i][0];
        }
        return result;
    }

    double norm() {
        double result = 0;
        for (int i = 0; i < height; i++) {
            result += this->table[i][0] * this->table[i][0];
        }
        return sqrt(result);
    }

    double getElement(int number) {
        return table[number][0];
    }

    void setElement(int index, double element) {
        table[index][0] = element;
        transposedTable[0][index] = element;
    }

    ColumnVector operator+(ColumnVector otherVector) {
        Matrix* thisVector = (Matrix*)this;
        Matrix result = thisVector->operator+((Matrix)otherVector);
        return ColumnVector(result);
    }

    ColumnVector operator-(ColumnVector otherVector) {
        Matrix* thisVector = (Matrix*)this;
        Matrix result = thisVector->operator-((Matrix)otherVector);
        return ColumnVector(result);
    }

    // NB: no multiplication overloading here is necessary.
    // (ColumnVector x RowVector) is represented by Matrix::operator*(a) with a = Matrix(1 x n).
    // (RowVector x ColumnVector) is represented by a.dot(b)
};

class AugmentedMatrix : public Matrix {
private:
    int stepNumber = 0;
    int permutationsOccurred = 0;

public:
    // Matrix [A|b] for solving SLAE via Gauss Elimination (LA Task 6)
    AugmentedMatrix(Matrix matrix, ColumnVector vector) : Matrix(matrix.getHeight(), matrix.getWidth()+1) {
        for (int rowIndex = 0; rowIndex < matrix.getHeight(); rowIndex++) {
            for (int columnIndex = 0; columnIndex < matrix.getWidth(); columnIndex++) {
                setElement(rowIndex, columnIndex, matrix.getElement(rowIndex, columnIndex));
            }
            setElement(rowIndex, matrix.getWidth(), vector.getElement(rowIndex));
        }
    }

    // Matrix [A|B]. When B = I, resulting [A|I] can be used for inversion procedure (LA Task 5)
    AugmentedMatrix(Matrix A, Matrix B) : Matrix(A.getHeight(), A.getWidth() + B.getWidth()) {
        if (A.getHeight() != B.getHeight()) {
            cout << "Error: the dimensional problem occurred" << endl;
        } else {
            for (int rowIndex = 0; rowIndex < A.getHeight(); rowIndex++) {
                for (int columnIndex = 0; columnIndex < A.getWidth(); columnIndex++) {
                    setElement(rowIndex, columnIndex, A.getElement(rowIndex, columnIndex));
                }
                for (int columnIndex = 0; columnIndex < B.getWidth(); columnIndex++) {
                    setElement(rowIndex, A.getWidth() + columnIndex, B.getElement(rowIndex, columnIndex));
                }
            }
        }
    }

    // Decorator pattern application (LA Task 4)
    explicit AugmentedMatrix(Matrix matrix) : Matrix(matrix.getHeight(), matrix.getWidth()) {
        this->table = matrix.getTable();
        this->transposedTable = matrix.getTransposedTable();
    }

private:
    // service method to eliminate the under-diagonal part of a single column
    AugmentedMatrix eliminateDown(Matrix result, int columnNumber, bool jointOutput, bool noStepNumbers) {
        // pivot check. Finding the absolute maximum pivot and exchange the rows
        int swapWithRow = columnNumber;
        double maxAbsolutePivot = abs(result.getElement(columnNumber, columnNumber));

        for (int rowIndex = columnNumber + 1; rowIndex < result.getHeight(); rowIndex++) {
            double absolutePivot = abs(result.getElement(rowIndex, columnNumber));
            if (absolutePivot > maxAbsolutePivot) {
                maxAbsolutePivot = absolutePivot;
                swapWithRow = rowIndex;
            }
        }
        // did the permutation occur?
        if (swapWithRow != columnNumber) {
            permutationsOccurred += 1;
            PermutationMatrix PIJ = PermutationMatrix(result.getHeight(), columnNumber, swapWithRow);
            result = PIJ*result;

            stepNumber += 1;
//            if (noStepNumbers) cout << "step: permutation" << endl;
//            else cout << "step #" << stepNumber << ": permutation" << endl;
//            if (jointOutput) {
//                cout << fixed << setprecision(2) << result;
//            } else {
//                separateOutput(AugmentedMatrix(result));
//            }
        }

        // forward elimination process
        for (int rowIndex = columnNumber + 1; rowIndex < result.getHeight(); rowIndex++) {
            double slot = result.getElement(rowIndex, columnNumber);
            if (slot == 0) continue; // skip the rest of the step if the slot is already zero

            EliminationMatrix EIJ = EliminationMatrix(result, rowIndex, columnNumber);
            result = EIJ*result;

            stepNumber += 1;
//            if (noStepNumbers) cout << "step: elimination" << endl;
//            else cout << "step #" << stepNumber << ": elimination" << endl;
//            if (jointOutput) {
//                cout << fixed << setprecision(2) << result;
//            } else {
//                separateOutput(AugmentedMatrix(result));
//            }
        }

        return AugmentedMatrix(result);
    }

    // Backward elimination function. Eliminates all the elements above the diagonal.
    AugmentedMatrix eliminateUp(Matrix result, int columnNumber, bool jointOutput) {
        for (int rowIndex = columnNumber - 1; rowIndex > -1; rowIndex--) {
            double slot = result.getElement(rowIndex, columnNumber);
            if (slot == 0) continue; // skip the rest of the step if the slot is already zero

            EliminationMatrix EIJ = EliminationMatrix(result, rowIndex, columnNumber);
            result = EIJ*result;

            stepNumber += 1;
//            cout << "step #" << stepNumber << ": elimination" << endl;
//            if (jointOutput) {
//                cout << fixed << setprecision(2) << result;
//            } else {
//                separateOutput(AugmentedMatrix(result));
//            }
        }
        return AugmentedMatrix(result);
    }

    AugmentedMatrix diagNorm(Matrix result) {
        // inversedDiagonal is D^-1, where D is taken from [D|DA^-1] after forward&backward elimination
        SquareMatrix inversedDiagonal = IdentityMatrix(result.getHeight());
        for (int i = 0; i < min(result.getWidth(), result.getHeight()); i++) {
            inversedDiagonal.setElement(i, i, 1 / result.getElement(i, i));
        }
        result = inversedDiagonal*result;
        return AugmentedMatrix(result);
    }

    // service tool for outputting [A|b] as separate A and b.
    void separateOutput(AugmentedMatrix A_b) {
        Matrix A = Matrix(A_b.getHeight(), A_b.getWidth() - 1);
        ColumnVector b = ColumnVector(A_b.getHeight());
        for (int rowIndex = 0; rowIndex < A_b.getHeight(); rowIndex++) {
            for (int columnIndex = 0; columnIndex < A_b.getWidth() - 1; columnIndex++) {
                A.setElement(rowIndex, columnIndex, A_b.getElement(rowIndex, columnIndex));
            }
            b.setElement(rowIndex, A_b.getElement(rowIndex, A_b.getWidth() - 1));
        }
        cout << fixed << setprecision(2) << A;
        cout << fixed << setprecision(2) << b;
    }

public:
    // LA-II Task 4. Outputs the whole determinant computation procedure. Returns det(matrix)
    // SSAD A4 Task A. Applies Strategy pattern via SSAD_moment to distinguish between LA and SSAD output requirements.
    double det(bool SSAD_moment) {
        if (getWidth() != getHeight()) {
            cout << "Error: the dimensional problem occurred" << endl;
            return 0; // if we add extra slots to make up for dimensional inequality, det(A) will be 0
        }

        stepNumber = 0;
        permutationsOccurred = 0;

        // forward elimination
        AugmentedMatrix refA = *this;
        for (int i = 0; i < min(getHeight(), getWidth()); i++) {
            refA = eliminateDown(refA, i, true, SSAD_moment);
        }
        // det(A) = (-1)^k * det(ref(A))
        // k = number of permutations; det(ref(A)) = product of diag. elements in ref(A)
        double determinant = 1;
        if (permutationsOccurred % 2 == 1) determinant = -1;
        for (int i = 0; i < min(getHeight(), getWidth()); i++) {
            determinant *= refA.getElement(i, i);
        }
        cout << "result:" << endl;
        cout << fixed << setprecision(2) << determinant << endl;
        return determinant;
    }

    // LA-II Task 5. Outputs the inverse procedure and returns A^-1
    // NB: There will be no output starting from LA-II Task 7.
    SquareMatrix inverse() {
        if (getWidth() != getHeight()) {
            cout << "Error: the dimensional problem occurred" << endl;
            return SquareMatrix(0); // no inverse is going to happen then
        }

        stepNumber = 0;
        permutationsOccurred = 0;

        // forward elimination
        AugmentedMatrix A_I = AugmentedMatrix(*this, IdentityMatrix(this->getHeight()));
        for (int i = 0; i < min(getHeight(), getWidth()); i++) {
            A_I = eliminateDown(A_I, i, true, false);
        }
        // backward elimination
        for (int i = min(getHeight(), getWidth()) - 1; i >= 0; i--) {
            A_I = eliminateUp(A_I, i, true);
        }
        // diagonal normalization
        A_I = diagNorm(A_I);
        // at this stage, A_I = [I|A^-1]. Let us extract the result then
        SquareMatrix inversedA = SquareMatrix(A_I.getHeight());
        for (int rowIndex = 0; rowIndex < A_I.getHeight(); rowIndex++) {
            for (int columnIndex = 0; columnIndex < A_I.getHeight(); columnIndex++) {
                double element = A_I.getElement(rowIndex, columnIndex + A_I.getHeight());
                inversedA.setElement(rowIndex, columnIndex, element);
            }
        }
        return inversedA;
    }

    // LA Task 6. Solves Ax = b given that it is presented in form of augmented matrix [A|b].
    // Returns column vector v, which is the solution for Ax = b.
    ColumnVector solveEquation() {
        if (getWidth() != getHeight() + 1) {
            cout << "Error: the dimensional problem occurred" << endl;
            return ColumnVector(0); // no solution is going to happen then
        }

        stepNumber = 0;
        permutationsOccurred = 0;

        // forward elimination
        AugmentedMatrix Ax_b = *this;
        cout << "step #0:" << endl;
        separateOutput(Ax_b);
        for (int i = 0; i < min(getHeight(), getWidth()); i++) {
            Ax_b = eliminateDown(Ax_b, i, false, false);
        }

        // backward elimination
        for (int i = min(getHeight(), getWidth()) - 1; i >= 0; i--) {
            Ax_b = eliminateUp(Ax_b, i, false);
        }

        // diagonal normalization
        cout << "Diagonal normalization:" << endl;
        Ax_b = diagNorm(Ax_b);
        separateOutput(Ax_b);

        // at this stage, Ax_b = [I|(A^-1)b]. Let us extract v = (A^-1)b then
        ColumnVector v = ColumnVector(Ax_b.getHeight());
        for (int rowIndex = 0; rowIndex < Ax_b.getHeight(); rowIndex++) {
            double element = Ax_b.getElement(rowIndex, Ax_b.getWidth() - 1);
            v.setElement(rowIndex, element);
        }
        cout << "result:" << endl;
        cout << fixed << setprecision(2) << v;
        return v;
    }

private:
    Matrix regressorMatrix(int polynomialDegree, ColumnVector t) {
        Matrix result = Matrix(t.getHeight(), polynomialDegree+1);
        for (int i = 0; i <= polynomialDegree; i++) {
            for (int j = 0; j < t.getHeight(); j++) {
                result.setElement(j, i, pow(t.getElement(j), i));
            }
        }
        return result;
    }

public:
    // LA Task 7. Fits polynomial to provided dataset t-to-b.
    ColumnVector LSA(int polynomialDegree) {
        ColumnVector b = ColumnVector(this->height);
        ColumnVector t = ColumnVector(this->height);
        for (int i = 0; i < this->height; i++) {
            t.setElement(i, this->getElement(i, 0));
            b.setElement(i, this->getElement(i, 1));
        }

        Matrix A = regressorMatrix(polynomialDegree, t);
        cout << "A:" << endl;
        cout << fixed << setprecision(4) << A;

        Matrix A_T_A = A.transpose()*A;
        cout << "A_T*A:" << endl;
        cout << fixed << setprecision(4) << A_T_A;

        Matrix A_T_A_inv = AugmentedMatrix(A_T_A).inverse();
        cout << "(A_T*A)^-1:" << endl;
        cout << fixed << setprecision(4) << A_T_A_inv;

        Matrix A_T = A.transpose();
        Matrix A_T_b = A_T*b;
        cout << "A_T*b:" << endl;
        cout << fixed << setprecision(4) << A_T_b;

        ColumnVector result = ColumnVector(A_T_A_inv*A_T_b);
        return result;
    }

    bool isDiagonalDominant() {
        for (int rowNumber = 0; rowNumber < this->height; rowNumber++) {
            double nonDiagonalSum = 0;
            for (int colNumber = 0; colNumber < this->width; colNumber++) {
                if (rowNumber != colNumber) nonDiagonalSum += table[rowNumber][colNumber];
            }
            if (nonDiagonalSum > table[rowNumber][rowNumber]) return false;
        }
        return true;
    }

    // LA Task 8. Solves Ax=b via Jacodi method
    ColumnVector solveJacobi(ColumnVector b, double epsilon) {
        SquareMatrix A = SquareMatrix(*this);
        SquareMatrix tau = IdentityMatrix(this->height);
        IdentityMatrix I = IdentityMatrix(this->height);
        for (int i = 0; i < tau.getHeight(); i++) {
            tau.setElement(i, i, 1 / getElement(i, i));
        }

        SquareMatrix alpha = SquareMatrix(I - tau*A);
        ColumnVector beta = ColumnVector(tau*b);
        cout << "alpha:" << endl;
        cout << fixed << setprecision(4) << alpha;
        cout << "beta:" << endl;
        cout << fixed << setprecision(4) << beta;

        ColumnVector x_i = beta;
        ColumnVector x_prev = beta;
        cout << "x(0):" << endl;
        cout << fixed << setprecision(4) << x_i;

        double e = epsilon + 1;

        for (int i = 1; e > epsilon; i++) {
            x_i = ColumnVector(alpha*x_prev + beta);
            e = ColumnVector(x_i-x_prev).norm();
            cout << "e: ";
            cout << fixed << setprecision(4) << e << endl;
            cout << "x(" << i << "):" << endl;
            cout << fixed << setprecision(4) << x_i;
            x_prev = x_i;
        }

        return x_i;
    }

    // LA Task 9. Solves Ax=b using Seidel method.
    ColumnVector solveSeidel(ColumnVector b, double epsilon) {
        SquareMatrix A = SquareMatrix(*this);
        SquareMatrix tau = IdentityMatrix(this->height);
        IdentityMatrix I = IdentityMatrix(this->height);
        for (int i = 0; i < tau.getHeight(); i++) {
            tau.setElement(i, i, 1 / getElement(i, i));
        }

        SquareMatrix alpha = SquareMatrix(I - tau*A);
        ColumnVector beta = ColumnVector(tau*b);

        cout << "beta:" << endl;
        cout << fixed << setprecision(4) << beta;
        cout << "alpha:" << endl;
        cout << fixed << setprecision(4) << alpha;

        SquareMatrix B = I - I;
        for (int i = 0; i < B.getHeight(); i++) {
            for (int j = 0; j < i; j++) {
                B.setElement(i, j, alpha.getElement(i, j));
            }
        }
        SquareMatrix C = alpha - B;
        SquareMatrix I_B_inv = AugmentedMatrix(I-B).inverse();

        cout << "B:" << endl;
        cout << fixed << setprecision(4) << B;
        cout << "C:" << endl;
        cout << fixed << setprecision(4) << C;
        cout << "I-B:" << endl;
        cout << fixed << setprecision(4) << (I-B);
        cout << "(I-B)_-1:" << endl;
        cout << fixed << setprecision(4) << I_B_inv;

        ColumnVector x_i = beta;
        ColumnVector x_prev = beta;
        cout << "x(0):" << endl;
        cout << fixed << setprecision(4) << x_i;

        double e = epsilon + 1;

        for (int i = 1; e > epsilon; i++) {
            x_i = ColumnVector(I_B_inv*C*x_prev + I_B_inv*beta);
            e = ColumnVector(x_i-x_prev).norm();
            cout << "e: ";
            cout << fixed << setprecision(4) << e << endl;
            cout << "x(" << i << "):" << endl;
            cout << fixed << setprecision(4) << x_i;
            x_prev = x_i;
        }

        return x_i;
    }
};


int LA_Task1() {
    int heightA, widthA, heightB, widthB, heightC, widthC;

    cin >> heightA >> widthA;
    Matrix A = Matrix(heightA, widthA);
    cin >> A;

    cin >> heightB >> widthB;
    Matrix B = Matrix(heightB, widthB);
    cin >> B;

    cin >> heightC >> widthC;
    Matrix C = Matrix(heightC, widthC);
    cin >> C;

    cout << A+B << B-A << C*A << A.transpose() << endl;

    return 0;
}

int LA_Task2() {
    int sideA, sideB, sideC;

    cin >> sideA;
    SquareMatrix A = SquareMatrix(sideA);
    cin >> A;

    cin >> sideB;
    SquareMatrix B = SquareMatrix(sideB);
    cin >> B;

    cin >> sideC;
    SquareMatrix C = SquareMatrix(sideC);
    cin >> C;

    cout << A+B << B-A << C*A << A.transpose() << endl;

    return 0;
}

int LA_Task3() {
    int sideA;
    int ELIM_I = 2, ELIM_J = 1;

    cin >> sideA;
    SquareMatrix A = SquareMatrix(sideA);
    cin >> A;

    EliminationMatrix E21 = EliminationMatrix(A, ELIM_I - 1, ELIM_J - 1);
    SquareMatrix B = E21*A;
    PermutationMatrix P21 = PermutationMatrix(sideA, ELIM_I - 1, ELIM_J - 1);
    SquareMatrix C = P21*A;

    cout << IdentityMatrix(3);
    cout << E21;
    cout << B;
    cout << P21;
    cout << C;

    return 0;
}

int LA_Task4() {
    int sideA;

    cin >> sideA;
    SquareMatrix A = SquareMatrix(sideA);
    cin >> A;

    AugmentedMatrix B = AugmentedMatrix(A);
    B.det(false);

    return 0;
}

int LA_Task5() {
    int sideA;

    cin >> sideA;
    SquareMatrix A = SquareMatrix(sideA);
    cin >> A;

    AugmentedMatrix B = AugmentedMatrix(A);
    B.inverse();

    return 0;
}

int LA_Task6() {
    int sideA;
    int lengthB;

    cin >> sideA;
    SquareMatrix A = SquareMatrix(sideA);
    cin >> A;

    cin >> lengthB;
    ColumnVector b = ColumnVector(lengthB);
    cin >> b;

    AugmentedMatrix Ax_b = AugmentedMatrix(A, b);
    Ax_b.solveEquation();

    return 0;
}

int SSAD_TaskA() {
    int sideA;

    cout << "Input the dimensions of a matrix:" << endl;
    cout << "n: ";
    cin >> sideA;

    cout << "Input n*n elements of a matrix:" << endl;
    SquareMatrix A = SquareMatrix(sideA);
    cin >> A;

    AugmentedMatrix B = AugmentedMatrix(A);
    B.det(true);

    return 0;
}

int LA_Task7() {
    int mLinesInDataset, nDegreesInPolynomial;
    cin >> mLinesInDataset;
    Matrix inputs = Matrix(mLinesInDataset, 2);
    cin >> inputs;
    cin >> nDegreesInPolynomial;
    AugmentedMatrix dataset = AugmentedMatrix(inputs);

    ColumnVector result = dataset.LSA(nDegreesInPolynomial);
    cout << "x~:" << endl;
    cout << fixed << setprecision(4) << result;

    return 0;
}

int LA_Task8() {
    int sideA, length_b;
    double eps;

    cin >> sideA;
    SquareMatrix inputA = SquareMatrix(sideA);
    cin >> inputA;

    cin >> length_b;
    ColumnVector b = ColumnVector(length_b);
    cin >> b;

    cin >> eps;

    AugmentedMatrix A = AugmentedMatrix(inputA);
    if (!A.isDiagonalDominant()) {
        cout << "The method is not applicable!" << endl;
        return 0;
    }

    A.solveJacobi(b, eps);

    return 0;
}

int LA_Task9() {
    int sideA, length_b;
    double eps;

    cin >> sideA;
    SquareMatrix inputA = SquareMatrix(sideA);
    cin >> inputA;

    cin >> length_b;
    ColumnVector b = ColumnVector(length_b);
    cin >> b;

    cin >> eps;

    AugmentedMatrix A = AugmentedMatrix(inputA);
    if (!A.isDiagonalDominant()) {
        cout << "The method is not applicable!" << endl;
        return 0;
    }

    A.solveSeidel(b, eps);

    return 0;
}

// Lotka-Volterra model output into the console
int LA_Task10() {
    double v0, k0, N;
    double a1, a2, b1, b2, T;
    cin >> v0 >> k0 >> a1 >> b1 >> a2 >> b2 >> T >> N;

    double equilibrium_v = a2 / b2;
    double equilibrium_k = a1 / b1;

    v0 -= equilibrium_v;
    k0 -= equilibrium_k;

    cout << "t:" << endl;
    for (int i = 0; i <= N; i++) {
        cout << fixed << setprecision(2) << i * (T / N) << " ";
    }
    cout << endl;

    cout << "v:" << endl;
    for (int i = 0; i <= N; i++) {
        double t = i * (T / N);
        double theta = t * pow(a1*a2, 0.5);
        double dv = v0 * cos(theta) - k0 * pow(a2/a1, 0.5) * (b1/b2) * sin(theta);
        cout << fixed << setprecision(2) << dv + equilibrium_v << " ";
    }
    cout << endl;

    cout << "k:" << endl;
    for (int i = 0; i <= N; i++) {
        double t = i * (T / N);
        double theta = t * pow(a1*a2, 0.5);
        double dk = v0 * pow(a1/a2, 0.5) * (b2/b1) * sin(theta) + k0 * cos(theta);
        cout << fixed << setprecision(2) << dk + equilibrium_k << " ";
    }
    cout << endl;
    return 0;
}

// GNUPlot orchestration (for LA-II Team only)
int LA_Plotting_LSA(Matrix dataset, ColumnVector coefficients) {
    Gnuplot gp;

    double min_x = numeric_limits<double>::max(), max_x = numeric_limits<double>::min();
    double min_y = numeric_limits<double>::max(), max_y = numeric_limits<double>::min();

    for (int i = 0; i < dataset.getHeight(); i++) {
        double x = round(dataset.getElement(i, 0) * 10000.0) / 10000.0;
        if (x > max_x) max_x = x;
        if (x < min_x) min_x = x;
        double y = round(dataset.getElement(i, 1) * 10000.0) / 10000.0;
        if (y > max_y) max_y = y;
        if (y < min_y) min_y = y;
    }

    min_x -= 2; min_y -= 2; max_x += 2; max_y += 2;
    gp << "set xrange [" << min_x << ":" << max_x << "]\n";
    gp << "set yrange [" << min_y << ":" << max_y << "]\n";

    // constructing polynomial line
    string polynomial = "";
    for (int i = 0; i < coefficients.getHeight(); i++) {
        double coefficient = coefficients.getElement(i);
        polynomial += "+"+to_string(coefficient)+"*x**"+to_string(i);
    }

    gp << "plot " + polynomial + " , '-' using 1:2 with points\n";
    for (int i = 0; i < dataset.getHeight(); i++) {
        double x = dataset.getElement(i, 0);
        double y = dataset.getElement(i, 1);
        gp << to_string(x)+"\t"+ to_string(y)+"\n";
    }

    return 0;
}

int assignment2Report() {
    int mLinesInDataset, nDegreesInPolynomial;
    cin >> mLinesInDataset;
    Matrix inputs = Matrix(mLinesInDataset, 2);
    cin >> inputs;
    cin >> nDegreesInPolynomial;
    AugmentedMatrix dataset = AugmentedMatrix(inputs);

    ColumnVector result = dataset.LSA(nDegreesInPolynomial);
    cout << "x~:" << endl;
    cout << fixed << setprecision(4) << result;

    return LA_Plotting_LSA(dataset, result);
}

// GNUPlot orchestration (for LA-II Team only)
// time_parametrized = true to plot v(t) and k(t)
// time_parametrized = false to plot v(k)
int LA_Plotting_Lotka_Volterra(double v0, double k0,
                               double a1, double b1,
                               double a2, double b2,
                               double T, double N,
                               bool time_parametrized) {

    double equilibrium_v = a2 / b2;
    double equilibrium_k = a1 / b1;

    v0 -= equilibrium_v;
    k0 -= equilibrium_k;

    vector<double> t, v, k;

    cout << "t:" << endl;
    for (int i = 0; i <= N; i++) {
        cout << fixed << setprecision(2) << i * (T / N) << " ";
        t.push_back(i * (T / N));
    }
    cout << endl;

    cout << "v:" << endl;
    for (int i = 0; i <= N; i++) {
        double t_i = i * (T / N);
        double theta = t_i * pow(a1*a2, 0.5);
        double dv = v0 * cos(theta) - k0 * pow(a2/a1, 0.5) * (b1/b2) * sin(theta);
        cout << fixed << setprecision(2) << dv + equilibrium_v << " ";
        v.push_back(dv + equilibrium_v);
    }
    cout << endl;

    cout << "k:" << endl;
    for (int i = 0; i <= N; i++) {
        double t_i = i * (T / N);
        double theta = t_i * pow(a1*a2, 0.5);
        double dk = v0 * pow(a1/a2, 0.5) * (b2/b1) * sin(theta) + k0 * cos(theta);
        cout << fixed << setprecision(2) << dk + equilibrium_k << " ";
        k.push_back(dk + equilibrium_k);
    }
    cout << endl;

    Gnuplot gp;

    double min_x = numeric_limits<double>::max(), max_x = numeric_limits<double>::min();
    double min_y = numeric_limits<double>::max(), max_y = numeric_limits<double>::min();

    if (time_parametrized) {
        for (int i = 0; i <= N; i++) {
            if (v[i] < min_y) min_y = v[i];
            if (v[i] > max_y) max_y = v[i];
            if (k[i] < min_y) min_y = k[i];
            if (k[i] > max_y) max_y = k[i];
        }

        min_x = 0; min_y -= 2; max_x = T; max_y += 2;
    } else {
        for (int i = 0; i <= N; i++) {
            if (v[i] < min_y) min_y = v[i];
            if (v[i] > max_y) max_y = v[i];
            if (k[i] < min_x) min_x = k[i];
            if (k[i] > max_x) max_x = k[i];
        }

        min_x -= 2; min_y -= 2; max_x += 2; max_y += 2;
    }

    gp << "set xrange [" << min_x << ":" << max_x << "]\n";
    gp << "set yrange [" << min_y << ":" << max_y << "]\n";

    if (time_parametrized) {
        gp << "plot '-' using 1:2 with lines linecolor rgb \"#00AA00\" title 'v(t)', ";
        gp << "'-' using 1:2 with lines linecolor rgb \"#AA0000\" title 'k(t)'\n";
        for (int i = 0; i <= N; i++) {
            double x = t[i];
            double y = v[i];
            gp << to_string(x)+"\t"+ to_string(y)+"\n";
        }

        gp << "e\n";

        for (int i = 0; i <= N; i++) {
            double x = t[i];
            double y = k[i];
            gp << to_string(x)+"\t"+ to_string(y)+"\n";
        }
    } else {
        gp << "plot '-' using 1:2 with lines linecolor rgb \"#666600\" title 'v(k)'\n";
        for (int i = 0; i <= N; i++) {
            double x = k[i];
            double y = v[i];
            gp << to_string(x)+"\t"+ to_string(y)+"\n";
        }
    }

    return 0;
}

int main() {
    double v0, k0, N;
    double a1, a2, b1, b2, T;
    cin >> v0 >> k0 >> a1 >> b1 >> a2 >> b2 >> T >> N;

    LA_Plotting_Lotka_Volterra(v0, k0, a1, b1, a2, b2, T, N, true);
    LA_Plotting_Lotka_Volterra(v0, k0, a1, b1, a2, b2, T, N, false);

    return 0;
}
