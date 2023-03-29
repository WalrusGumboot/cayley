/*! `cayley` is a crate for generic linear algebra. It aims to do everything stack-allocated
and constantly sized (though there are workarounds possible if dynamically sized
types are needed). `cayley` is named after Arthur Cayley, a prominent mathematician who
introduced matrix multiplication.

In addition to this, it aims to assume as little as possible
about the type over which its structures are generic. For example, you can construct
an identity matrix of any type that implements `One`, `Zero` and `Copy`, and you can multiply
matrices of different types A and B, so long as there exists a type C so that A * B = C
and C + C = C. In practice, of course, all numerical types meet these conditions.

Due to the nature of generic matrices, it's necessary to use the `#[feature(generic_const_exprs)]`
feature; there is no other way to provide compile-time multiplicability or invertibility checks.

*/

#![allow(dead_code, incomplete_features)]
#![doc(test(attr(feature(generic_const_exprs))))]
#![feature(generic_const_exprs)]
#![deny(missing_docs)]
use num_traits::{NumOps, One, Signed, Zero};
use std::fmt::{self, Debug, Display};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

/// The following is some weird shit. This enum is generic over a boolean condition.
/// It then only implements the IsTrue trait for `DimensionAssertion<true>`, so that
/// an assertion can be made within a function signature or an impl block.
pub enum DimensionAssertion<const CONDITION: bool> {}
/// IsTrue is only ever implemented on `DimensionAssertion<true>`. See its documentation
/// for info on why this exists.
pub trait IsTrue {}
impl IsTrue for DimensionAssertion<true> {}

/// The base Matrix struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix<T, const N: usize, const M: usize>
where
    [(); N * M]:,
{
    /// The array representing the data. The items are sorted in 'reading order', that is:
    /// from left to right, next from top to bottom. It's generally not recommended to
    /// access this array directly. Instead, indexing into the matrix using a tuple of `usize`s.
    pub data: [T; N * M],
    /// The number of rows in the matrix. Note that though this is typechecked
    /// it is often still useful to just get the number of rows in a matrix by accessing
    /// a field.
    pub rows: usize,
    /// The number of columns in the matrix. Note that though this is typechecked
    /// it is often still useful to just get the number of columns in a matrix by accessing
    /// a field.
    pub cols: usize,
}

/// Convenience stuff.
impl<T, const N: usize, const M: usize> Index<(usize, usize)> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(
            index.0 < self.rows,
            "Tried indexing into row {}, which outside of the matrix (has {} rows).",
            index.0,
            self.rows
        );
        assert!(
            index.1 < self.cols,
            "Tried indexing into column {}, which outside of the matrix (has {} column).",
            index.1,
            self.cols
        );
        &self.data[index.0 * self.cols + index.1]
    }
}

impl<T, const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

impl<T, const N: usize, const M: usize> Display for Matrix<T, N, M>
where
    T: Display,
    [(); N * M]:,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Displaying a matrix is kind of interesting: we want to have nicely
        // spaced and nicely aligned numbers, but use cases might arise where
        // the elements of a Matrix implement NumOps and Display but aren't
        // numbers in and of themselves. We have to figure out the longest
        // string representation first, then do all of the printing stuff.

        let string_reps = self.data.iter().map(|e| e.to_string()).collect::<Vec<_>>();
        let longest = string_reps.iter().fold(0, |current_max, new| {
            if new.len() > current_max {
                new.len()
            } else {
                current_max
            }
        });

        let padded_string_reps = string_reps
            .iter()
            .map(|s| format!("{:0l$} ", s, l = longest))
            .collect::<Vec<String>>();

        for row in padded_string_reps.chunks_exact(self.cols) {
            writeln!(
                f,
                "{}",
                row.iter().fold(String::new(), |mut acc, val| {
                    acc.push_str(val);
                    acc
                })
            )?;
        }

        Ok(())
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    [(); N * M]:,
{
    /// Checks if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Executes a closure returning a boolean on every item in the matrix. Returns `true`
    /// if the closure returned true for all elements of the matrix.
    pub fn all<F>(&self, func: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        self.data.iter().all(func)
    }

    /// Executes a closure returning a boolean on every item in the matrix. Returns `true`
    /// if the closure returned true for any element of the matrix.
    pub fn any<F>(&self, func: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        self.data.iter().any(func)
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    T: Copy,
    [(); N * M]:,
{
    /// Returns an array containing the values of a given row.
    pub fn row(&self, r: usize) -> [T; M] {
        assert!(
            r < N,
            "Index out of bounds: specified row {r} but matrix only has {N} rows."
        );
        let mut row = [self[(0, 0)]; M];
        for i in 0..M {
            row[i] = self[(r, i)];
        }
        row
    }

    /// Returns an array containing the values of a given column.
    pub fn col(&self, c: usize) -> [T; N] {
        assert!(
            c < M,
            "Index out of bounds: specified column {c} but matrix only has {M} columns."
        );
        let mut col = [self[(0, 0)]; N];
        for i in 0..N {
            col[i] = self[(i, c)];
        }
        col
    }
}

impl<T, const N: usize, const M: usize> From<Vec<Vec<T>>> for Matrix<T, N, M>
where
    T: Copy,
    [(); N * M]:,
{
    fn from(value: Vec<Vec<T>>) -> Matrix<T, N, M> {
        assert!(
            value.iter().all(|row| row.len() == value[0].len()),
            "Not all rows have the same length."
        );
        assert_eq!(value.len(), N, "Dimensionality does not hold: specified a vector with {} elements but Matrix type is supposed to have {N} rows.", value.len());
        assert_eq!(value[0].len(), M, "Dimensionality does not hold: specified a vector whose elements have {} elements but Matrix type is supposed to have {M} columns.", value[0].len());

        let mut data = [value[0][0]; N * M];
        let mut flattened = value.iter().flatten();
        for i in 0..N * M {
            data[i] = *flattened.next().unwrap();
        }
        Self {
            data,
            rows: value.len(),
            cols: value[0].len(),
        }
    }
}

/// Constructors.
impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    T: Zero + Copy,
    [(); N * M]:,
{
    /// Initialises a matrix filled with zeroes.
    ///
    /// This requires the type of the matrix to implement the num_traits::Zero trait.
    /// A type implementing NumAssignOps but not Zero is very rare though.
    ///
    /// ## Panics
    ///
    /// If the specified rows and columns don't create a matrix with N elements.
    pub fn zeroes(r: usize, c: usize) -> Self {
        assert_eq!(
            N, r,
            "Dimensionality of the matrix does not hold: rows do not match."
        );
        assert_eq!(
            M, c,
            "Dimensionality of the matrix does not hold: columns do not match."
        );

        Matrix {
            data: [T::zero(); N * M],
            rows: r,
            cols: c,
        }
    }

    /// Constructs a Matrix from a closure. The closure takes in two zero-indexed usizes and
    /// returns any type T. It requires that T implements Zero for allocation purposes.
    /// This should probably be changed to implementing Default, now that I think about it.
    ///
    /// ## Example
    ///
    /// ```
    /// use cayley::Matrix;
    /// let m: Matrix<usize, 2, 3> = Matrix::from_closure(2, 3, |x, y| x + y);
    /// assert_eq!(m, Matrix::from(vec![vec![0, 1, 2], vec![1, 2, 3]]));
    /// ```
    pub fn from_closure<F>(r: usize, c: usize, func: F) -> Self
    where
        F: Fn(usize, usize) -> T,
    {
        let mut result = Matrix::zeroes(r, c);
        for x in 0..N {
            for y in 0..M {
                result[(x, y)] = func(x, y);
            }
        }

        result
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    T: One + Copy,
    [(); N * M]:,
{
    /// Initialises a matrix filled with ones.
    ///
    /// This requires the type of the matrix to implement the num_traits::One trait.
    /// A type implementing NumAssignOps but not One is very rare though.
    ///
    /// ## Panics
    ///
    /// If the specified rows and columns don't create a matrix with N elements.
    pub fn ones(r: usize, c: usize) -> Self {
        assert_eq!(
            N, r,
            "Dimensionality of the matrix does not hold: rows do not match."
        );
        assert_eq!(
            M, c,
            "Dimensionality of the matrix does not hold: columns do not match."
        );

        Matrix {
            data: [T::one(); N * M],
            rows: r,
            cols: c,
        }
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: Zero + One + Copy,
    [(); N * N]:,
{
    /// Constructs the identity matrix. Requires the target type to implement
    /// Zero and One, for obvious reasons.
    ///
    /// ## Example
    /// ```
    /// use cayley::Matrix;
    /// let m: Matrix<i32, 3, 3> = Matrix::identity(3);
    /// assert_eq!(
    ///     m,
    ///     Matrix::from(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]])
    /// );
    /// ```
    pub fn identity(size: usize) -> Self {
        let mut base = Matrix::zeroes(size, size);
        for i in 0..size {
            base[(i, i)] = T::one();
        }
        base
    }
}

// Addition.
impl<T, Q, const N: usize, const M: usize> Add<Matrix<Q, N, M>> for Matrix<T, N, M>
where
    T: Add<Q, Output = T> + Copy,
    Q: Copy,
    [(); N * M]:,
{
    type Output = Matrix<T, N, M>;
    fn add(self, rhs: Matrix<Q, N, M>) -> Self::Output {
        assert_eq!(
            self.rows, rhs.rows,
            "Matrices do not have the same dimension."
        );
        let mut data: [T; N * M] = self.data;
        for i in 0..N * M {
            data[i] = data[i] + rhs.data[i];
        }

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T, Q, const N: usize, const M: usize> AddAssign<Matrix<Q, N, M>> for Matrix<T, N, M>
where
    T: AddAssign<Q>,
    Q: Copy,
    [(); N * M]:,
{
    fn add_assign(&mut self, rhs: Matrix<Q, N, M>) {
        for i in 0..N * M {
            self.data[i] += rhs.data[i];
        }
    }
}

// Subtraction.
impl<T, Q, const N: usize, const M: usize> Sub<Matrix<Q, N, M>> for Matrix<T, N, M>
where
    T: Sub<Q, Output = T> + Copy,
    Q: Copy,
    [(); N * M]:,
{
    type Output = Matrix<T, N, M>;
    fn sub(self, rhs: Matrix<Q, N, M>) -> Self::Output {
        assert_eq!(
            self.rows, rhs.rows,
            "Matrices do not have the same dimension."
        );
        let mut data: [T; N * M] = self.data;
        for i in 0..N * M {
            data[i] = data[i] - rhs.data[i];
        }

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T, Q, const N: usize, const M: usize> SubAssign<Matrix<Q, N, M>> for Matrix<T, N, M>
where
    T: SubAssign<Q>,
    Q: Copy,
    [(); N * M]:,
{
    fn sub_assign(&mut self, rhs: Matrix<Q, N, M>) {
        for i in 0..N * M {
            self.data[i] -= rhs.data[i];
        }
    }
}

// Negation

impl<T, const N: usize, const M: usize> Neg for Matrix<T, N, M>
where
    T: Zero + Copy + Neg<Output = T>,
    [(); N * M]:,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut data = Matrix::zeroes(N, M);
        for x in 0..N {
            for y in 0..M {
                data[(x, y)] = -self[(x, y)];
            }
        }

        data
    }
}

// Multiplication

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    T: Copy + Mul<Output = T> + Zero,
    [(); N * M]:,
{
    /// Multiplies every element of a Matrix with a scalar value.
    pub fn scalar_mul(&self, rhs: T) -> Matrix<T, N, M> {
        let mut data: Matrix<T, N, M> = Matrix::zeroes(N, M);

        for x in 0..N {
            for y in 0..M {
                data[(x, y)] = self[(x, y)].mul(rhs);
            }
        }

        data
    }
}

impl<T, Q, R, const N: usize, const M: usize, const O: usize, const P: usize> Mul<Matrix<Q, O, P>>
    for Matrix<T, N, M>
where
    T: Copy + Mul<Q, Output = R>,
    Q: Copy,
    R: Add + Zero + Copy,
    [(); N * M]:,
    [(); O * P]:,
    [(); N * P]:,
    DimensionAssertion<{ M == O }>: IsTrue,
{
    type Output = Matrix<R, N, P>;
    /// Multiplies two matrices.
    ///
    /// ## Examples
    ///
    /// ```compile_fail
    /// let m1: Matrix<i32, 2, 3> = Matrix::from(vec![vec![1, 2, 3], vec![4, 5, 6]]);
    /// let m2: Matrix<i32, 2, 2> = Matrix::from(vec![vec![1, 2], vec![3, 4]]);
    /// let a = m1 * m2; // this does not compile!
    /// ```
    fn mul(self, rhs: Matrix<Q, O, P>) -> Self::Output {
        let mut result: Matrix<R, N, P> = Matrix::zeroes(N, P);

        for x in 0..N {
            for y in 0..P {
                let mut dot_product_terms = [R::zero(); M];
                for i in 0..M {
                    dot_product_terms[i] = self[(x, i)] * rhs[(i, y)];
                }
                result[(x, y)] = dot_product_terms
                    .iter()
                    .fold(R::zero(), |acc, val| acc + *val);
            }
        }

        result
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    [(); N * M]:,
    [(); M * N]:,
    T: Copy + Zero,
{
    /// Returns the transpose of the matrix.
    ///
    /// Note that this means that (unless the Matrix is square) the return type is
    /// different from the caller type: `Matrix<u8, 2, 3>.transpose()` returns a `Matrix<u8, 3, 2>`.
    pub fn transpose(&self) -> Matrix<T, M, N> {
        let mut result = Matrix::zeroes(M, N);

        for x in 0..M {
            for y in 0..N {
                result[(x, y)] = self[(y, x)];
            }
        }

        result
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    [(); N * M]:,
    [(); (N - 1) * (M - 1)]:,
    T: Zero + Copy,
{
    /// Calculates the submatrix of a matrix. The submatrix is the smaller matrix aqcuired from
    /// ignoring the existence of one row and one column from that matrix.
    pub fn submatrix(&self, r: usize, c: usize) -> Matrix<T, { N - 1 }, { M - 1 }> {
        assert!(r < self.rows, "Specified out-of-bounds index in creating a submatrix: indexed row {} while matrix has {} rows.", r, self.rows);
        assert!(c < self.cols, "Specified out-of-bounds index in creating a submatrix: indexed column {} while matrix has {} columns.", c, self.cols);

        let mut subm: Matrix<T, { N - 1 }, { M - 1 }> =
            Matrix::zeroes(self.rows - 1, self.cols - 1);

        let mut x_counter = 0usize;
        let mut y_counter = 0usize;

        for x in 0..N {
            if x == r {
                continue;
            }
            for y in 0..M {
                if y == c {
                    continue;
                }
                subm[(x_counter, y_counter)] = self[(x, y)];
                y_counter += 1;
            }
            y_counter = 0;
            x_counter += 1;
        }

        subm
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    [(); (N - 1) * (N - 1)]:,
    T: Zero + Copy,
{
    /// Calculates the submatrix of a square matrix. This implementation is necessary for some
    /// other calculations to be valid (e.g. the comatrix); it does not perform better at all.
    pub fn square_submatrix(&self, r: usize, c: usize) -> Matrix<T, { N - 1 }, { N - 1 }> {
        self.submatrix(r, c)
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    [(); N * M]:,
    T: Zero + One + Copy + NumOps + PartialEq + PartialOrd + Signed,
{
    /// Applies an elementary row operation to a specified row.
    ///
    /// This function takes a source row index and a target row index, as
    /// well as a closure that itself takes two inputs. These two inputs act
    /// as a placeholder for an element of the source and target row respectively
    /// in an expression describing what happens to each element of the target row.
    /// For example, the operation "add two times the first row to the fifth row" would
    /// look like `matrix.row_op(0, 4, |r1, r2| 2 * r1 + r2)`.
    ///
    /// Note that this makes `row_op` quite a powerful function: a row effectively be
    /// copied to another one, or set to a custom value.
    pub fn row_op<F>(&mut self, source_row: usize, target_row: usize, func: F)
    where
        F: Fn(T, T) -> T,
    {
        assert!(
            source_row < N,
            "Index out of bounds: tried to access row {source_row} but matrix has {N} rows."
        );
        assert!(
            target_row < N,
            "Index out of bounds: tried to access row {target_row} but matrix has {N} rows."
        );
        for i in 0..M {
            self[(target_row, i)] = func(self[(source_row, i)], self[(target_row, i)]);
        }
    }

    /// Swaps two rows. Mutates the matrix in-place.
    pub fn row_swap(&mut self, r1: usize, r2: usize) {
        assert!(
            r1 < N,
            "Index out of bounds: tried to access row {r1} but matrix has {N} rows."
        );
        assert!(
            r2 < N,
            "Index out of bounds: tried to access row {r2} but matrix has {N} rows."
        );

        let tmp = self.row(r1);
        for i in 0..N {
            self[(r1, i)] = self[(r2, i)];
            self[(r2, i)] = tmp[i];
        }
    }

    /// Mutates a matrix to be in row echelon form. The naming
    /// convention here is slightly annoying as "row echelon form" would
    /// be shortened to 'ref', which is obviously a reserved Rust keyword.
    ///
    /// Unlike most methods implemented on `Matrix`, this mutates `self` without
    /// returning a new matrix. What it does return, however, is a product of all
    /// row operation factors. This is used to calculate the determinant.
    pub fn row_ef(&mut self) -> T {
        // the accumulating factor
        let mut accum = T::one();

        if self.is_in_row_echelon_form() {
            return accum;
        }

        // We operate column by column or row by row depending on which is the bigger one
        // but frankly, it doesn't matter. Most matrices have more columns than rows though
        // so the variable is named column

        for current_column in 0..M.min(N) {
            // If this column consists only of zeroes, it "accounts for a redundant
            // variable" if you're speaking in terms of systems of linear equations.
            // Hence, it can safely be ignored.

            if self.col(current_column).iter().all(|e| e.is_zero()) {
                continue;
            }

            // To get zeroes in all rows below this one, we first calculate an
            // intermediate row. This row is used to subtract from per row,
            // such that the current row gets a zero.
            //
            // If the pivot is zero at this point, there will be a zero division
            // in the following algorithm. However, because of the `continue` statement
            // above, it is guaranteed that there is at least one nonzero element in
            // this column. So, we swap the row containing that element and the current one
            // around.
            // Note that this shouldn't check entries in the column that come before the
            // currently checked row! Those might be nonzero, but are irrelevant here.

            if self[(current_column, current_column)].is_zero() {
                let index_of_nonzero = self
                    .col(current_column)
                    .iter()
                    .enumerate()
                    .skip(current_column) // to ignore previous possibly nonzero values
                    .find(|(_idx, elem)| !elem.is_zero())
                    .unwrap()
                    .0;

                self.row_swap(current_column, index_of_nonzero);
                accum = accum.neg(); // swapping two rows multiplies the determinant by negative one
            }

            let pivot = self[(current_column, current_column)];

            //println!("before:\n{self}");

            //println!("on column {current_column}. pivot is {pivot}");

            for row_below in (current_column + 1)..N {
                let factor = self[(row_below, current_column)] / pivot;

                //println!("making a zero in row {row_below}. factor is {factor}");

                if factor.is_zero() {
                    continue;
                }

                let intermediate_row: Vec<T> = self
                    .row(current_column)
                    .iter()
                    .map(|e| *e * factor)
                    .collect();

                //println!("intermediate row: {intermediate_row:?}");

                for i in 0..M {
                    self[(row_below, i)] = self[(row_below, i)] - intermediate_row[i];
                }
            }

            //println!("{self}");

            accum = accum * self[(current_column, current_column)];
        }

        accum
    }

    /// Checks if the matrix is in row echelon form.
    pub fn is_in_row_echelon_form(&self) -> bool {
        if self == &Matrix::zeroes(N, M) {
            return true;
        }
        let mut rows = (0..N).map(|i| self.row(i)).peekable();
        let mut leading_zeroes = -1isize;
        for _i in 0..N {
            let r = rows.next().unwrap();
            // println!(row is {r:#?}");
            if r.iter().all(|e| e.is_zero()) {
                // println!("all zeroes");
                match rows.peek() {
                    None => {
                        // println!("and we're the last");
                        return true;
                    } // we are the last row
                    Some(next_row) => {
                        // println!("we're not the last");
                        if next_row.iter().all(|e| e.is_zero()) {
                            // println!("the next row is all zeroes so we continue");
                            continue;
                        } else {
                            // println!("the next row is not all zeroes so no row ech form");
                            return false;
                        }
                    }
                }
            }
            // the row should contain more than leading_zeroes zeroes
            // before anything else appears in the row (it's safe to unwrap here
            // since we checked for all-zero rows above)
            let first_nonzero_entry = r.iter().position(|e| !e.is_zero()).unwrap() as isize;

            // println!("our first nonzero entry is {first_nonzero_entry}");

            if first_nonzero_entry > leading_zeroes {
                // println!("which is greater than our last number of leading zeroes :)");
                leading_zeroes = first_nonzero_entry;
            } else {
                // println!("which is not greater than our last number of leading zeroes");
                return false;
            }
        }

        true
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    T: Zero + One + Copy + NumOps + Signed + PartialEq + PartialOrd,
    [(); (N - 1) * (N - 1)]:,
{
    /// Calculates the matrix of cofactors.
    pub fn comatrix(&self) -> Self {
        let mut result: Matrix<T, N, N> = Matrix::zeroes(N, N);

        for x in 0..N {
            for y in 0..N {
                result[(x, y)] = self.square_submatrix(x, y).determinant() * {
                    if (x + y) % 2usize == 0 {
                        T::one()
                    } else {
                        T::one().neg()
                    }
                };
            }
        }

        result
    }

    /// Calculates the adjugate (also known as the classical adjoint) of a square matrix.
    pub fn adjugate(&self) -> Self {
        self.comatrix().transpose()
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    T: Copy + NumOps + Zero + One + PartialEq + PartialOrd + Signed,
{
    /// Calculates the determinant of a Matrix.
    /// Requires the relevant type to implement NumOps (Add, Sub, Mul, Div), as well
    /// as Copy, Zero, One and Neg.
    pub fn determinant(&self) -> T {
        match N {
            1 => self[(0, 0)],
            2 => self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)],
            3 => {
                self[(0, 0)] * self[(1, 1)] * self[(2, 2)]
                    + self[(0, 1)] * self[(1, 2)] * self[(2, 0)]
                    + self[(0, 2)] * self[(1, 0)] * self[(2, 1)]
                    - self[(0, 2)] * self[(1, 1)] * self[(2, 0)]
                    - self[(0, 1)] * self[(1, 0)] * self[(2, 2)]
                    - self[(0, 0)] * self[(1, 2)] * self[(2, 1)]
            }
            _ => {
                

                self.clone().row_ef()
            }
        }
    }

    /// Tries to decompose the matrix into a lower triangular and upper triangular matrix
    /// according to Crout's method. Returns an optional tuple `Some((lower, upper))`.
    /// The implementation is taken straight from Wikipedia:
    /// https://en.wikipedia.org/w/index.php?title=Crout_matrix_decomposition&oldid=956132782
    pub fn crout_decomposition(self) -> Option<(Self, Self)> {
        let mut lower: Matrix<T, N, N> = Matrix::zeroes(N, N);
        let mut upper: Matrix<T, N, N> = Matrix::zeroes(N, N);

        for i in 0..N {
            upper[(i, i)] = T::one();
        }

        for j in 0..N {
            for i in 0..N {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + lower[(i, k)] * upper[(k, j)];
                }
                lower[(i, j)] = self[(i, j)] - sum;
            }

            for i in j..N {
                let mut sum = T::zero();
                for k in 0..j {
                    sum = sum + lower[(j, k)] * upper[(k, i)];
                }
                if lower[(j, j)].is_zero() {
                    // The determinant is zero, and hence this decomposition cannot be returned.
                    return None;
                }
                upper[(j, i)] = (self[(j, i)] - sum) / lower[(j, j)];
            }
        }

        Some((lower, upper))
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    [(); (N - 1) * (N - 1)]:,
    T: Copy + NumOps + Zero + PartialEq + One + Signed + PartialEq + PartialOrd,
{
    /// Attempts to calculate the inverse of the Matrix. Note that this is only
    /// implemented for `Matrix<T, N, N>`, i.e. square matrices.
    ///
    /// ## Returns
    ///
    /// An `Option<Self>`: `None` if the matrix isn't invertible and `Some(m)` with
    /// m being the inverted matrix.
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == T::zero() {
            None
        } else {
            Some(self.adjugate().scalar_mul(T::one() / det))
        }
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    T: Copy + Zero + PartialEq,
{
    /// Checks if a square matrix is symmetric.
    pub fn is_symmetric(&self) -> bool {
        self.transpose() == *self
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    T: Copy + Zero + PartialEq + Neg<Output = T>,
{
    /// Checks if a square matrix is skew-symmetric.
    pub fn is_skew_symmetric(&self) -> bool {
        self.transpose() == -*self
    }
}

mod tests;
