#![allow(dead_code)]
use num_traits::{NumAssignOps, One, Zero};
use std::fmt::{self, Display};
use std::ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign};

/// The base Matrix struct.
pub struct Matrix<T, const N: usize> {
    data: [T; N],
    rows: usize,
    cols: usize,
}

/// Convenience stuff.
impl<T, const N: usize> Index<(usize, usize)> for Matrix<T, N> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.rows + index.1]
    }
}

impl<T, const N: usize> IndexMut<(usize, usize)> for Matrix<T, N> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.rows + index.1]
    }
}

impl<T, const N: usize> Display for Matrix<T, N>
where
    T: Display,
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

/// Constructors.
impl<T, const N: usize> Matrix<T, N>
where
    T: Zero + Copy,
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
        assert_eq!(N, r * c, "Dimensionality of the matrix does not hold.");
        Matrix {
            data: [T::zero(); N],
            rows: r,
            cols: c,
        }
    }
}

impl<T, const N: usize> Matrix<T, N>
where
    T: One + Copy,
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
        assert_eq!(N, r * c, "Dimensionality of the matrix does not hold.");
        Matrix {
            data: [T::one(); N],
            rows: r,
            cols: c,
        }
    }
}

impl<T, const N: usize> Matrix<T, N>
where
    T: Zero + One + Copy,
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
impl<T, Q, const N: usize> Add<Matrix<Q, N>> for Matrix<T, N>
where
    T: Add<Q, Output = T> + Copy,
    Q: Copy,
{
    type Output = Matrix<T, N>;
    fn add(self, rhs: Matrix<Q, N>) -> Self::Output {
        assert_eq!(
            self.rows, rhs.rows,
            "Matrices do not have the same dimension."
        );
        let mut data: [T; N] = self.data;
        for i in 0..N {
            data[i] = data[i] + rhs.data[i];
        }

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T, Q, const N: usize> AddAssign<Matrix<Q, N>> for Matrix<T, N>
where
    T: AddAssign<Q>,
    Q: Copy,
{
    fn add_assign(&mut self, rhs: Matrix<Q, N>) {
        for i in 0..N {
            self.data[i] += rhs.data[i];
        }
    }
}

// Subtraction.
impl<T, Q, const N: usize> Sub<Matrix<Q, N>> for Matrix<T, N>
where
    T: Sub<Q, Output = T> + Copy,
    Q: Copy,
{
    type Output = Matrix<T, N>;
    fn sub(self, rhs: Matrix<Q, N>) -> Self::Output {
        assert_eq!(
            self.rows, rhs.rows,
            "Matrices do not have the same dimension."
        );
        let mut data: [T; N] = self.data;
        for i in 0..N {
            data[i] = data[i] - rhs.data[i];
        }

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T, Q, const N: usize> SubAssign<Matrix<Q, N>> for Matrix<T, N>
where
    T: SubAssign<Q>,
    Q: Copy,
{
    fn sub_assign(&mut self, rhs: Matrix<Q, N>) {
        for i in 0..N {
            self.data[i] -= rhs.data[i];
        }
    }
}

mod tests;
