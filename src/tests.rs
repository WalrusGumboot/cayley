#[cfg(test)]
pub mod tests {
    use crate::Matrix;

    #[test]
    fn create_identity_matrix() {
        let m: Matrix<i32, 9> = Matrix::identity(3);
        assert_eq!(m, Matrix::from(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]))
    }

    #[test]
    fn uint_matrix_addition() {
        let m1: Matrix<u8, 9> = Matrix::identity(3);
        let m2: Matrix<u8, 9> = Matrix::ones(3, 3);

        assert_eq!(m1 + m2, Matrix::from(vec![vec![2, 1, 1], vec![1, 2, 1], vec![1, 1, 2]]))
    }

    #[test]
    fn generic_addition() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct A(i32);
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        struct B(i32);

        impl std::ops::Add for A {
            type Output = A;
            fn add(self, rhs: Self) -> Self::Output {
                A(self.0 + rhs.0)
            }
        }

        impl std::ops::Mul for A {
            type Output = A;
            fn mul(self, rhs: Self) -> Self::Output {
                A(self.0 * rhs.0)
            }
        }

        impl num_traits::One for A {
            fn one() -> Self {
                A(1)
            }

            fn is_one(&self) -> bool
            where
                Self: PartialEq,
            {
                self.0 == 1
            }

            fn set_one(&mut self) {
                *self = A(1);
            }
        }

        impl num_traits::Zero for A {
            fn zero() -> Self {
                A(0)
            }

            fn is_zero(&self) -> bool {
                self.0 == 0
            }

            fn set_zero(&mut self) {
                *self = A(0);
            }
        }

        impl std::ops::Mul for B {
            type Output = B;
            fn mul(self, rhs: Self) -> Self::Output {
                B(self.0 * rhs.0)
            }
        }

        impl num_traits::One for B {
            fn one() -> Self {
                B(1)
            }

            fn is_one(&self) -> bool
            where
                Self: PartialEq,
            {
                self.0 == 1
            }

            fn set_one(&mut self) {
                *self = B(1);
            }
        }

        impl std::ops::Add<B> for A {
            type Output = A;
            fn add(self, rhs: B) -> Self::Output {
                A(self.0 + rhs.0)
            }
        }

        let m1: Matrix<A, 9> = Matrix::identity(3);
        let m2: Matrix<B, 9> = Matrix::ones(3, 3);

        let a = m1 + m2; // type of a is Matrix<A, 16>
        assert_eq!(a, Matrix::from(vec![vec![A(2), A(1), A(1)], vec![A(1), A(2), A(1)], vec![A(1), A(1), A(2)]]))
    }
}
