#[cfg(test)]
pub mod tests {
    use crate::Matrix;

    #[test]
    fn create_identity_matrix() {
        let m: Matrix<f64, 16> = Matrix::identity(4);
        println!("{m}")
    }

    #[test]
    fn uint_matrix_addition() {
        let m1: Matrix<u8, 16> = Matrix::identity(4);
        let m2: Matrix<u8, 16> = Matrix::ones(4, 4);

        println!("{}", m1 + m2);
    }

    #[test]
    fn generic_addition() {
        #[derive(Clone, Copy, PartialEq, Eq)]
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

        let m1: Matrix<A, 16> = Matrix::identity(4);
        let m2: Matrix<B, 16> = Matrix::ones(4, 4);

        let a = m1 + m2; // this works
    }
}
