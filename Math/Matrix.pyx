import copy
import random
import math
from Math.MatrixNotPositiveDefinite import MatrixNotPositiveDefinite
from Math.MatrixNotSquare import MatrixNotSquare
from Math.Eigenvector import Eigenvector
from Math.MatrixDimensionMismatch import MatrixDimensionMismatch
from Math.Vector cimport Vector
from Math.MatrixRowMismatch import MatrixRowMismatch
from Math.DeterminantZero import DeterminantZero
from Math.MatrixRowColumnMismatch import MatrixRowColumnMismatch
from Math.MatrixColumnMismatch import MatrixColumnMismatch
from Math.MatrixNotSymmetric import MatrixNotSymmetric


cdef class Matrix(object):


    cpdef constructor1(self, int row):
        cdef int i
        self.__row = row
        self.__col = row
        self.initZeros()
        for i in range(self.__row):
            self.__values[i][i] = 1.0

    cpdef constructor2(self, int row, int col):
        self.__row = row
        self.__col = col
        self.initZeros()

    cpdef constructor3(self, int row, int col, float minValue):
        cdef int i
        self.__row = row
        self.__col = col
        self.initZeros()
        for i in range(self.__row):
            self.__values[i][i] = minValue

    cpdef constructor4(self, int row, int col, float minValue, float maxValue, int seed):
        self.__row = row
        self.__col = col
        random.seed(seed)
        self.__values = [[random.uniform(minValue, maxValue) for _ in range(self.__col)] for _ in
                         range(self.__row)]

    cpdef constructor5(self, Vector row, Vector col):
        cdef int i, j
        self.__row = row.size()
        self.__col = col.size()
        self.initZeros()
        for i in range(row.size()):
            for j in range(col.size()):
                self.__values[i][j] = row.getValue(i) * col.getValue(j)

    def __init__(self,
                 row,
                 col=None,
                 minValue=None,
                 maxValue=None,
                 seed=None):
        """
        Constructor of Matrix class which takes row and column numbers (Vectors) as inputs.

        PARAMETERS
        ----------
        row : int (or Vector)
            is used to create matrix.
        col : int (or Vector)
            is used to create matrix.
        minValue : double
            minimum Value for the initialization
        maxValue : double
            maximum Value for the initialization
        seed : int
            seed for the random
        """
        if isinstance(row, int):
            if col is not None:
                if minValue is None:
                    self.constructor2(row, col)
                elif maxValue is None:
                    self.constructor3(row, col, minValue)
                else:
                    self.constructor4(row, col, minValue, maxValue, seed)
            else:
                self.constructor1(row)
        elif isinstance(row, Vector) and isinstance(col, Vector):
            self.constructor5(row, col)

    cpdef initZeros(self):
        """
        Initializes the matrix values with 0.
        :return: Initialized matrix.
        """
        cdef int i, j
        self.__values = [[0 for i in range(self.__col)] for j in range(self.__row)]

    cpdef Matrix clone(self):
        """
        Clones the matrix.
        :return: Deep cloned copy.
        """
        return copy.deepcopy(self)

    cpdef double getValue(self,
                          int rowNo,
                          int colNo):
        """
        The getter for the index at given rowNo and colNo of values list.

        PARAMETERS
        ----------
        rowNo : int
            integer input for row number.
        colNo : int
            integer input for column number.

        RETURNS
        -------
        double
            item at given index of values list.
        """
        return self.__values[rowNo][colNo]

    cpdef setValue(self,
                   int rowNo,
                   int colNo,
                   double value):
        """
        The setter for the value at given index of values list.

        PARAMETERS
        ----------
        rowNo : int
            integer input for row number.
        colNo : int
            integer input for column number.
        value : double
            is used to set at given index.
        """
        self.__values[rowNo][colNo] = value

    cpdef addValue(self,
                   int rowNo,
                   int colNo,
                   double value):
        """
        The addValue method adds the given value to the item at given index of values list.

        PARAMETERS
        ----------
        rowNo : int
            integer input for row number.
        colNo : int
            integer input for column number.
        value : double
            is used to add to given item at given index.
        """
        self.__values[rowNo][colNo] += value

    cpdef increment(self,
                    int rowNo,
                    int colNo):
        """
        The increment method adds 1 to the item at given index of values list.

        PARAMETERS
        ----------
        rowNo : int
            integer input for row number.
        colNo : int
            integer input for column number.
        """
        self.__values[rowNo][colNo] += 1

    cpdef int getRow(self):
        """
        The getter for the row variable.

        RETURNS
        -------
        int
            row number.
        """
        return self.__row

    cpdef Vector getRowVector(self, int row):
        """
        The getRowVector method returns the vector of values list at given row input.

        PARAMETERS
        ----------
        row : int
            row integer input for row number.

        RETURNS
        -------
        Vector
            Vector of values list at given row input.
        """
        cdef list rowList
        cdef Vector rowVector
        rowList = self.__values[row]
        rowVector = Vector(rowList)
        return rowVector

    cpdef int getColumn(self):
        """
        The getter for the col variable.

        RETURNS
        -------
        int
            column number.
        """
        return self.__col

    cpdef list getColumnVector(self, int column):
        """
         * The getColumnVector method creates a Vector and adds items at given column number of values list
         * to the Vector.

        PARAMETERS
        ----------
        column : int
            column integer input for column number.

        RETURNS
        -------
        Vector
            Vector of given column number.
        """
        cdef list columnVector
        cdef int i
        columnVector = []
        for i in range(self.__row):
            columnVector.append(self.__values[i][column])
        return columnVector

    cpdef columnWiseNormalize(self):
        """
        The columnWiseNormalize method, first accumulates items column by column then divides items
        by the summation.
        """
        cdef int i
        for i in range(self.__row):
            total = sum(self.__values[i])
            self.__values[i][:] = [x / total for x in self.__values[i]]

    cpdef multiplyWithConstant(self, double constant):
        """
        The multiplyWithConstant method takes a constant as an input and multiplies each item of values list
        with given constant.

        PARAMETERS
        ----------
        constant : double
            constant value to multiply items of values list.
        """
        cdef int i
        for i in range(self.__row):
            self.__values[i][:] = [x * constant for x in self.__values[i]]

    cpdef divideByConstant(self, double constant):
        """
        The divideByConstant method takes a constant as an input and divides each item of values list
        with given constant.

        PARAMETERS
        ----------
        constant : double
            constant value to divide items of values list.
        """
        cdef int i
        for i in range(self.__row):
            self.__values[i][:] = [x / constant for x in self.__values[i]]

    cpdef add(self, Matrix m):
        """
        The add method takes a Matrix as an input and accumulates values list with the
        corresponding items of given Matrix. If the sizes of both Matrix and values list do not match,
        it throws MatrixDimensionMismatch exception.

        PARAMETERS
        ----------
        m : Matrix
            Matrix type input.
        """
        cdef int i, j
        if self.__row != m.__row or self.__col != m.__col:
            raise MatrixDimensionMismatch
        for i in range(self.__row):
            for j in range(self.__col):
                self.__values[i][j] += m.__values[i][j]

    cpdef addRowVector(self,
                       int rowNo,
                       Vector v):
        """
        The add method which takes a row number and a Vector as inputs. It sums up the corresponding values at the given
        row of values list and given Vector. If the sizes of both Matrix and values list do not match, it throws
        MatrixColumnMismatch exception.

        PARAMETERS
        ----------
        rowNo : int
            integer input for row number.
        v : Vector
            Vector type input.
        """
        cdef int i
        if self.__col != v.size():
            raise MatrixColumnMismatch
        for i in range(self.__col):
            self.__values[rowNo][i] += v.getValue(i)

    cpdef subtract(self, Matrix m):
        """
        The subtract method takes a Matrix as an input and subtracts from values list the
        corresponding items of given Matrix. If the sizes of both Matrix and values list do not match,
        it throws {@link MatrixDimensionMismatch} exception.

        PARAMETERS
        ----------
        m : Matrix
            Matrix type input.
        """
        cpdef int i, j
        if self.__row != m.__row or self.__col != m.__col:
            raise MatrixDimensionMismatch
        for i in range(self.__row):
            for j in range(self.__col):
                self.__values[i][j] -= m.__values[i][j]

    cpdef Vector multiplyWithVectorFromLeft(self, Vector v):
        """
        The multiplyWithVectorFromLeft method takes a Vector as an input and creates a result list.
        Then, multiplies values of input Vector starting from the left side with the values list,
        accumulates the multiplication, and assigns to the result list. If the sizes of both Vector
        and row number do not match, it throws MatrixRowMismatch exception.

        PARAMETERS
        ----------
        v : Vector
            Vector type input.

        RETURNS
        -------
        Vector
            Vector that holds the result.
        """
        cdef int i, j
        cdef double total
        cdef Vector result
        if self.__row != v.size():
            raise MatrixRowMismatch
        result = Vector()
        for i in range(self.__col):
            total = 0.0
            for j in range(self.__row):
                total += v.getValue(j) * self.__values[j][i]
            result.add(total)
        return result

    cpdef Vector multiplyWithVectorFromRight(self, Vector v):
        """
        The multiplyWithVectorFromRight method takes a Vector as an input and creates a result list.
        Then, multiplies values of input Vector starting from the right side with the values list,
        accumulates the multiplication, and assigns to the result list. If the sizes of both Vector
        and row number do not match, it throws MatrixColumnMismatch exception.

        PARAMETERS
        ----------
        v : Vector
            Vector type input.

        RETURNS
        -------
        Vector
            Vector that holds the result.
        """
        cdef int i, j
        cdef double total
        cdef Vector result
        if self.__col != v.size():
            raise MatrixColumnMismatch
        result = Vector()
        for i in range(self.__row):
            total = 0.0
            for j in range(self.__col):
                total += v.getValue(j) * self.__values[i][j]
            result.add(total)
        return result

    cpdef double columnSum(self, int columnNo):
        """
        The columnSum method takes a column number as an input and accumulates items at given column number of values
        list.

        PARAMETERS
        ----------
        columnNo : int
            Column number input.

        RETURNS
        -------
        double
            summation of given column of values list.
        """
        cdef double total
        cdef int i
        total = 0
        for i in range(self.__row):
            total += self.__values[i][columnNo]
        return total

    cpdef Vector sumOfRows(self):
        """
        The sumOfRows method creates a mew result Vector and adds the result of columnDum method's corresponding
        index to the newly created result Vector.

        RETURNS
        -------
        Vector
            Vector that holds column sum.
        """
        cdef Vector result
        cdef int i
        result = Vector()
        for i in range(self.__col):
            result.add(self.columnSum(i))
        return result

    cpdef double rowSum(self, int rowNo):
        """
        The rowSum method takes a row number as an input and accumulates items at given row number of values list.

         * @param rowNo Row number input.
         * @return summation of given row of values {@link java.lang.reflect.Array}.
        """
        return sum(self.__values[rowNo])

    cpdef Matrix multiply(self, Matrix m):
        """
        The multiply method takes a Matrix as an input. First it creates a result Matrix and puts the
        accumulatated multiplication of values list and given Matrix into result
        Matrix. If the size of Matrix's row size and values list's column size do not match,
        it throws MatrixRowColumnMismatch exception.

        PARAMETERS
        ----------
        m : Matrix
            Matrix type input.

        RETURNS
        -------
        Matrix
            result Matrix.
        """
        cdef Matrix result
        cdef int i, j, k
        cdef double total
        if self.__col != m.__row:
            raise MatrixRowColumnMismatch
        result = Matrix(self.__row, m.__col)
        for i in range(self.__row):
            for j in range(m.__col):
                total = 0.0
                for k in range(self.__col):
                    total += self.__values[i][k] * m.__values[k][j]
                result.__values[i][j] = total
        return result

    cpdef Matrix elementProduct(self, Matrix m):
        """
        The elementProduct method takes a Matrix as an input and performs element wise multiplication. Puts result
        to the newly created Matrix. If the size of Matrix's row and column size does not match with the values
        list's row and column size, it throws MatrixDimensionMismatch exception.

        PARAMETERS
        ----------
        m : Matrix
            Matrix type input.

        RETURNS
        -------
        Matrix
            result Matrix.
        """
        cdef Matrix result
        cdef int i, j
        if self.__row != m.__row or self.__col != m.__col:
            raise MatrixDimensionMismatch
        result = Matrix(self.__row, self.__col)
        for i in range(self.__row):
            for j in range(self.__col):
                result.__values[i][j] = self.__values[i][j] * m.__values[i][j]
        return result

    cpdef double sumOfElements(self):
        """
        The sumOfElements method accumulates all the items in values list and
        returns this summation.

        RETURNS
        -------
        double
            sum of the items of values list.
        """
        cdef double total
        cdef int i
        total = 0.0
        for i in range(self.__row):
            total += sum(self.__values[i])
        return total

    cpdef double trace(self):
        """
        The trace method accumulates items of values list at the diagonal.

        RETURNS
        -------
        double
            sum of items at diagonal.
        """
        cdef double total
        cdef int i
        if self.__row != self.__col:
            raise MatrixNotSquare
        total = 0.0
        for i in range(self.__row):
            total += self.__values[i][i]
        return total

    cpdef Matrix transpose(self):
        """
        The transpose method creates a new Matrix, then takes the transpose of values list
        and puts transposition to the Matrix.

        RETURNS
        -------
        Matrix
            Matrix type output.
        """
        cdef Matrix result
        cdef int i, j
        result = Matrix(self.__col, self.__row)
        for i in range(self.__row):
            for j in range(self.__col):
                result.__values[j][i] = self.__values[i][j]
        return result

    cpdef Matrix partial(self,
                         int rowStart,
                         int rowEnd,
                         int colStart,
                         int colEnd):
        """
        The partial method takes 4 integer inputs; rowStart, rowEnd, colStart, colEnd and creates a Matrix size of
        rowEnd - rowStart + 1 x colEnd - colStart + 1. Then, puts corresponding items of values list
        to the new result Matrix.

        PARAMETERS
        ----------
        rowStart : int
            integer input for defining starting index of row.
        rowEnd : int
            integer input for defining ending index of row.
        colStart : int
            integer input for defining starting index of column.
        colEnd : int
            integer input for defining ending index of column.

        RETURNS
        -------
        Matrix
            result Matrix.
        """
        cdef Matrix result
        cdef int i, j
        result = Matrix(rowEnd - rowStart + 1, colEnd - colStart + 1)
        for i in range(rowStart, rowEnd + 1):
            for j in range(colStart, colEnd + 1):
                result.__values[i - rowStart][j - colStart] = self.__values[i][j]
        return result

    cpdef bint isSymmetric(self):
        """
        The isSymmetric method compares each item of values list at positions (i, j) with (j, i)
        and returns true if they are equal, false otherwise.

        RETURNS
        -------
        bool
            true if items are equal, false otherwise.
        """
        cdef int i, j
        if self.__row != self.__col:
            raise MatrixNotSquare
        for i in range(self.__row - 1):
            for j in range(self.__row):
                if self.__values[i][j] != self.__values[j][i]:
                    return False
        return True

    cpdef double determinant(self):
        """
        The determinant method first creates a new list, and copies the items of  values
        list into new list. Then, calculates the determinant of this
        new list.

        RETURNS
        -------
        double
            determinant of values list.
        """
        cdef double det
        cdef Matrix copyOfMatrix
        cdef int i, j, k
        if self.__row != self.__col:
            raise MatrixNotSquare
        det = 1.0
        copyOfMatrix = copy.deepcopy(self)
        for i in range(self.__row):
            det *= copyOfMatrix.__values[i][i]
            if det == 0.0:
                break
            for j in range(i + 1, self.__row):
                ratio = copyOfMatrix.__values[j][i] / copyOfMatrix.__values[i][i]
                for k in range(i, self.__col):
                    copyOfMatrix.__values[j][k] = copyOfMatrix.__values[j][k] - copyOfMatrix.__values[i][k] * ratio
        return det

    cpdef inverse(self):
        """
        The inverse method finds the inverse of values list.
        """
        cdef Matrix b
        cdef int i, j, k, l, ll, irow, icol
        cdef list indxc, indxr, ipiv
        cdef double big, dum, pivinv
        if self.__row != self.__col:
            raise MatrixNotSquare
        b = Matrix(self.__row, self.__row)
        indxc = []
        indxr = []
        ipiv = []
        for j in range(self.__row):
            ipiv.append(0)
        for i in range(1, self.__row + 1):
            big = 0.0
            irow = -1
            icol = -1
            for j in range(1, self.__row + 1):
                if ipiv[j - 1] != 1:
                    for k in range(1, self.__row + 1):
                        if ipiv[k - 1] == 0:
                            if abs(self.__values[j - 1][k - 1]) >= big:
                                big = abs(self.__values[j - 1][k - 1])
                                irow = j
                                icol = k
            if irow == -1 or icol == -1:
                raise DeterminantZero
            ipiv[icol - 1] = ipiv[icol - 1] + 1
            if irow != icol:
                for l in range(1, self.__row + 1):
                    dum = self.__values[irow - 1][l - 1]
                    self.__values[irow - 1][l - 1] = self.__values[icol - 1][l - 1]
                    self.__values[icol - 1][l - 1] = dum
                for l in range(1, self.__row + 1):
                    dum = b.__values[irow - 1][l - 1]
                    b.__values[irow - 1][l - 1] = b.__values[icol - 1][l - 1]
                    b.__values[icol - 1][l - 1] = dum
            indxr.append(irow)
            indxc.append(icol)
            if self.__values[icol - 1][icol - 1] == 0:
                raise DeterminantZero
            pivinv = 1.0 / self.__values[icol - 1][icol - 1]
            self.__values[icol - 1][icol - 1] = 1.0
            for l in range(1, self.__row + 1):
                self.__values[icol - 1][l - 1] = self.__values[icol - 1][l - 1] * pivinv
            for l in range(1, self.__row + 1):
                b.__values[icol - 1][l - 1] = b.__values[icol - 1][l - 1] * pivinv
            for ll in range(1, self.__row + 1):
                if ll != icol:
                    dum = self.__values[ll - 1][icol - 1]
                    self.__values[ll - 1][icol - 1] = 0.0
                    for l in range(1, self.__row + 1):
                        self.__values[ll - 1][l - 1] = self.__values[ll - 1][l - 1] - self.__values[icol - 1][
                            l - 1] * dum
                    for l in range(1, self.__row + 1):
                        b.__values[ll - 1][l - 1] = b.__values[ll - 1][l - 1] - b.__values[icol - 1][l - 1] * dum
        for l in range(self.__row, 0, -1):
            if indxr[l - 1] != indxc[l - 1]:
                for k in range(1, self.__row + 1):
                    dum = self.__values[k - 1][indxr[l - 1] - 1]
                    self.__values[k - 1][indxr[l - 1] - 1] = self.__values[k - 1][indxc[l - 1] - 1]
                    self.__values[k - 1][indxc[l - 1] - 1] = dum

    cpdef Matrix choleskyDecomposition(self):
        """
        The choleskyDecomposition method creates a new Matrix and puts the Cholesky Decomposition of values Array
        into this Matrix. Also, it throws MatrixNotSymmetric exception if it is not symmetric and
        MatrixNotPositiveDefinite exception if the summation is negative.

        RETURNS
        -------
        Matrix
            Matrix type output.
        """
        cdef Matrix b
        cdef int i, j
        cdef double total
        if not self.isSymmetric():
            raise MatrixNotSymmetric
        b = Matrix(self.__row, self.__col)
        for i in range(self.__row):
            for j in range(i, self.__row):
                total = self.__values[i][j]
                for k in range(i - 1, -1, -1):
                    total -= self.__values[i][k] * self.__values[j][k]
                if i == j:
                    if total <= 0.0:
                        raise MatrixNotPositiveDefinite
                    b.__values[i][i] = math.sqrt(total)
                else:
                    b.__values[j][i] = total / b.__values[i][i]
        return b

    cpdef __rotate(self,
                   double s,
                   double tau,
                   int i,
                   int j,
                   int k,
                   int l):
        """
        The rotate method rotates values list according to given inputs.

        PARAMETERS
        ----------
        s : double
            double input.
        tau : double
            double input.
        i : int
            integer input.
        j : int
            integer input.
        k : int
            integer input.
        l : int
            integer input.
        """
        cdef double g, h
        g = self.__values[i][j]
        h = self.__values[k][l]
        self.__values[i][j] = g - s * (h + g * tau)
        self.__values[k][l] = h + s * (g - h * tau)

    def characteristics(self)-> list:
        """
        The characteristics method finds and returns a sorted list of Eigenvecto}s. And it throws
        MatrixNotSymmetric exception if it is not symmetric.

        RETURNS
        -------
        list
            A sorted list of Eigenvectors.
        """
        cdef Matrix matrix1, v
        cdef list d, b, z, result
        cdef double EPS, sm, threshold, g, h, t, theta, c, s, tau
        cdef int ip, i, iq
        if not self.isSymmetric():
            raise MatrixNotSymmetric
        matrix1 = copy.deepcopy(self)
        v = Matrix(self.__row, self.__row, 1.0)
        d = []
        b = []
        z = []
        EPS = 0.000000000000000001
        for ip in range(self.__row):
            b.append(matrix1.__values[ip][ip])
            d.append(matrix1.__values[ip][ip])
            z.append(0.0)
        for i in range(1, 51):
            sm = 0.0
            for ip in range(self.__row - 1):
                for iq in range(ip + 1, self.__row):
                    sm += abs(matrix1.__values[ip][iq])
            if sm == 0.0:
                break
            if i < 4:
                threshold = 0.2 * sm / (self.__row ** 2)
            else:
                threshold = 0.0
            for ip in range(self.__row - 1):
                for iq in range(ip + 1, self.__row):
                    g = 100.0 * abs(matrix1.__values[ip][iq])
                    if i > 4 and g <= EPS * abs(d[ip]) and g <= EPS * abs(d[iq]):
                        matrix1.__values[ip][iq] = 0.0
                    else:
                        if abs(matrix1.__values[ip][iq]) > threshold:
                            h = d[iq] - d[ip]
                            if g <= EPS * abs(h):
                                t = matrix1.__values[ip][iq] / h
                            else:
                                theta = 0.5 * h / matrix1.__values[ip][iq]
                                t = 1.0 / (abs(theta) + math.sqrt(1.0 + theta ** 2))
                                if theta < 0.0:
                                    t = -t
                            c = 1.0 / math.sqrt(1 + t ** 2)
                            s = t * c
                            tau = s / (1.0 + c)
                            h = t * matrix1.__values[ip][iq]
                            z[ip] -= h
                            z[iq] += h
                            d[ip] -= h
                            d[iq] += h
                            matrix1.__values[ip][iq] = 0.0
                            for j in range(ip):
                                matrix1.__rotate(s, tau, j, ip, j, iq)
                            for j in range(ip + 1, iq):
                                matrix1.__rotate(s, tau, ip, j, j, iq)
                            for j in range(iq + 1, self.__row):
                                matrix1.__rotate(s, tau, ip, j, iq, j)
                            for j in range(self.__row):
                                v.__rotate(s, tau, j, ip, j, iq)
            for ip in range(self.__row):
                b[ip] = b[ip] + z[ip]
                d[ip] = b[ip]
                z[ip] = 0.0
        result = []
        for i in range(self.__row):
            if d[i] > 0:
                result.append(Eigenvector(d[i], v.getColumnVector(i)))
        result.sort(key=lambda eigenvector: eigenvector.getEigenvalue(), reverse=True)
        return result

    def __repr__(self):
        return f"{self.__values}"
