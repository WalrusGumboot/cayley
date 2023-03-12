#![allow(dead_code)]
#![feature(generic_const_exprs)]
#![deny(missing_docs)]
use num_traits::{NumOps, One, Zero};
use std::fmt::{self, Display};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

/// The following is some weird shit.
pub enum DimensionAssertion<const CONDITION: bool> {}
pub trait IsTrue {}
impl IsTrue for DimensionAssertion<true> {}

/// The base Matrix struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Matrix<T, const N: usize, const M: usize>
where
    [(); N * M]:,
{
    data: [T; N * M],
    rows: usize,
    cols: usize,
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
        // the elements of a Matrix implement NumAssignOps and Display but aren't
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

        for row in padded_string_reps.chunks_exact(self.rows) {
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

impl<T, const N: usize, const M: usize> Matrix<T, N, M>
where
    T: Zero + One + Copy,
    [(); N * M]:,
    DimensionAssertion<{ N == M }>: IsTrue,
{
    pub fn identity(size: usize) -> Self {
        let mut base = Matrix::zeroes(size, size);
        for i in 0..size {
            base[(i, i)] = T::one();
        }
        base
    }
}

/// Operations on matrices.
/// Note that the resulting matrix takes on the type of the left matrix.

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

// Multiplication

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

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    T: Copy + NumOps + Zero,
{
    pub fn determinant(&self) -> T {
        match N {
            1 => self[(0, 0)],
            2 => self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)],
            _ => todo!(),
        }
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    [(); N * N]:,
    T: Copy + NumOps + Zero + PartialEq,
{
    pub fn inverse(&self) -> Option<Self> {
        if self.determinant() == T::zero() {
            None
        } else {
            todo!()
        }
    }
}

mod tests;
