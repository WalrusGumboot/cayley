#[cfg(test)]
pub mod tests {
    use num_traits::Signed;

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
    fn get_column() {
        let m: Matrix<usize, 3, 3> = Matrix::from_closure(3, 3, |x, y| 3 * x + y);
        assert_eq!(m.col(2), [2, 5, 8])
    }

    #[test]
    fn get_row() {
        let m: Matrix<usize, 3, 3> = Matrix::from_closure(3, 3, |x, y| 3 * x + y);
        assert_eq!(m.row(2), [6, 7, 8])
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
        let m2: Matrix<f64, 3, 3> = Matrix::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        let m3: Matrix<f64, 3, 3> = Matrix::from(vec![
            vec![1.0, 2.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let m4: Matrix<f64, 4, 4> = Matrix::from(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 5.0, 6.0, 1.0],
            vec![2.0, 3.0, 5.0, 4.0],
            vec![3.0, 5.0, 8.0, 6.0],
        ]);

        let m5: Matrix<f64, 4, 4> = Matrix::from(vec![
            vec![-1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![3.0, 1.0, 2.0, 3.0],
            vec![5.0, 2.0, -4.0, -1.0],
        ]);

        assert_eq!(m1.determinant(), -2.0_f64);
        assert_eq!(m2.determinant(), 0.0_f64);
        assert_eq!(m3.determinant(), 9.0_f64);
        assert_eq!(m4.determinant(), 6.0_f64);
        assert_eq!(m5.determinant(), -198.0_f64);
    }

    #[test]
    fn adjugate() {
        let m: Matrix<f64, 3, 3> = Matrix::from(vec![
            vec![2.0, -3.0, 4.0],
            vec![1.0, 0.0, 5.0],
            vec![1.0, 1.0, 9.0],
        ]);

        assert_eq!(
            m.adjugate(),
            Matrix::from(vec![
                vec![-5.0, 31.0, -15.0],
                vec![-4.0, 14.0, -6.0],
                vec![1.0, -5.0, 3.0]
            ])
        );
    }

    #[test]
    fn inverse() {
        let m1: Matrix<f64, 2, 2> = Matrix::from(vec![vec![1.0, 2.0], vec![0.0, 0.0]]);
        assert_eq!(m1.inverse(), None);

        let m2: Matrix<f64, 3, 3> = Matrix::identity(3);
        assert_eq!(m2.inverse(), Some(m2));

        let m3: Matrix<f64, 2, 2> = Matrix::from(vec![vec![4.0, 7.0], vec![2.0, 6.0]])
            .inverse()
            .unwrap(); // has nonzero determinant
        let m3_inverse: Matrix<f64, 2, 2> = Matrix::from(vec![vec![0.6, -0.7], vec![-0.2, 0.4]]);
        assert!(m3
            .data
            .iter()
            .zip(m3_inverse.data.iter())
            .all(|(a, b)| a.abs() - b.abs() < 1e-10));
    }

    #[test]
    fn any_and_all() {
        let m: Matrix<u8, 4, 4> = Matrix::from_closure(4, 4, |x, y| (2 * x + 4 * y) as u8);

        assert!(m.all(|n| n % 2 == 0));
        assert!(m.any(|n| n == &2));
    }

    #[test]
    fn creation_from_closure() {
        let m: Matrix<usize, 2, 3> = Matrix::from_closure(2, 3, |x, y| x + y);
        assert_eq!(m, Matrix::from(vec![vec![0, 1, 2], vec![1, 2, 3]]));
    }

    #[test]
    fn submatrix() {
        let m: Matrix<usize, 4, 4> = Matrix::from_closure(4, 4, |x, y| 4 * x + y);
        assert_eq!(
            m.submatrix(3, 3),
            Matrix::from(vec![vec![0, 1, 2], vec![4, 5, 6], vec![8, 9, 10]])
        )
    }

    #[test]
    fn row_ops() {
        let mut m: Matrix<i32, 3, 3> = Matrix::from_closure(3, 3, |x, y| (3 * x + y) as i32);

        m.row_op(0, 1, |r1, r2| 2 * r1 + r2);

        assert_eq!(
            m,
            Matrix::from(vec![vec![0, 1, 2], vec![3, 6, 9], vec![6, 7, 8]])
        );
    }

    #[test]
    fn is_row_echelon_form() {
        assert!(Matrix::<i32, 5, 5>::identity(5).is_in_row_echelon_form());
        assert!(Matrix::<i32, 5, 5>::zeroes(5, 5).is_in_row_echelon_form());
        assert!(!Matrix::<i32, 5, 5>::ones(5, 5).is_in_row_echelon_form());
        assert!(Matrix::<i32, 3, 5>::from(vec![
            vec![0, 2, 1, 3, 4],
            vec![0, 0, 0, 3, 2],
            vec![0, 0, 0, 0, 0]
        ])
        .is_in_row_echelon_form());
    }

    #[test]
    fn row_echelon_form() {
        let mut m = Matrix::<f64, 4, 5>::from(vec![
            vec![2.0, 0.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 3.0, 1.0],
            vec![4.0, 1.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 2.0, 3.0, 8.0],
        ]);

        m.row_ef();
        assert!(m.is_in_row_echelon_form())
    }
}
