#[cfg(test)]
pub mod tests {
    use crate::Matrix;

    #[test]
    fn create_identity_matrix() {
        let m: Matrix<i32, 3, 3> = Matrix::identity(3);
        assert_eq!(
            m,
            Matrix::from(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]])
        )
    }

    #[test]
    fn uint_matrix_addition() {
        let m1: Matrix<u8, 3, 3> = Matrix::identity(3);
        let m2: Matrix<u8, 3, 3> = Matrix::ones(3, 3);

        assert_eq!(
            m1 + m2,
            Matrix::from(vec![vec![2, 1, 1], vec![1, 2, 1], vec![1, 1, 2]])
        )
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

        let m1: Matrix<A, 3, 3> = Matrix::identity(3);
        let m2: Matrix<B, 3, 3> = Matrix::ones(3, 3);

        let a = m1 + m2; // type of a is Matrix<A, 16>
        assert_eq!(
            a,
            Matrix::from(vec![
                vec![A(2), A(1), A(1)],
                vec![A(1), A(2), A(1)],
                vec![A(1), A(1), A(2)]
            ])
        )
    }

    #[test]
    fn matrix_multiplication() {
        let m1: Matrix<i32, 2, 3> = Matrix::from(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let m2: Matrix<i32, 3, 2> = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);

        assert_eq!(m1 * m2, Matrix::from(vec![vec![22, 28], vec![49, 64]]));
    }

    #[test]
    #[ignore = "This function should not be able to compile."]
    fn invalid_matrix_multiplication() {
        let m1: Matrix<i32, 2, 3> = Matrix::from(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let m2: Matrix<i32, 2, 2> = Matrix::from(vec![vec![1, 2], vec![3, 4]]);

        // let _ = m1 * m2; this errors!
    }

    #[test]
    fn transposition() {
        let m: Matrix<i32, 2, 3> = Matrix::from(vec![vec![1, 2, 3], vec![4, 5, 6]]);

        assert_eq!(
            m.transpose(),
            Matrix::from(vec![vec![1, 4], vec![2, 5], vec![3, 6]])
        );

        // ∀ A: (Aᵀ)ᵀ = A
        assert_eq!(m, m.transpose().transpose());

        let id: Matrix<i32, 8, 8> = Matrix::identity(8);

        // ∀ n: Iₙ = (Iₙ)ᵀ
        assert_eq!(id.transpose(), id);
    }

    #[test]
    fn determinant() {
        let m1: Matrix<f64, 2, 2> = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        assert_eq!(m1.determinant(), -2.0f64);
    }

    #[test]
    fn inverse() {
        let m1: Matrix<f64, 2, 2> = Matrix::from(vec![vec![1.0, 2.0], vec![0.0, 0.0]]);

        assert_eq!(m1.inverse(), None);
    }

    #[test]
    fn creation_from_closure() {
        let m: Matrix<usize, 2, 3> = Matrix::from_closure(2, 3, |x, y| x + y);
        assert_eq!(m, Matrix::from(vec![vec![0, 1, 2], vec![1, 2, 3]]));
    }
}
