# cayley - generic, stack-allocated linear algebra in Rust

cayley is a crate that fills a small niche: it provides a ***generic matrix type that allocates its data on the stack***.
This makes cayley pretty quick even on very large scales. It also handles slightly more complex operations on matrices,
such as finding the determinant and inverse of a matrix. These calculations are not very optimised as of now, but at least
they exist. cayley is named after Arthur Cayley, a prominent mathematician who introduced matrix multiplication.

One of the major features of this crate (and at the time of writing its greatest drawback too) is that it uses the
`generic_const_exprs` feature. This is experimental Rust, but it allows a *type-safe dimensionality system*. Take, for example,
matrix multiplication. A 4 by 3 matrix can't be multiplied by a 7 by 5 one. I wasn't able to find any other crate that's able
to check that **at compile time**. cayley does. It's impossible to compile code that will try to perform a matrix operation which
shoudln't be possible. The hindrance is that Rust's type inference doesn't fully support generic constant expressions (yet) and
as a result the user has to explicitly annotate the dimensions of the Matrix type, like so:

``` rust
use cayley::Matrix;

let m1: Matrix<f64, 2, 3> = Matrix::ones(2, 3);
let m2: Matrix<f64, 3, 2> = Matrix::from_closure(3, 2, |x, y| (x + y) as f64);

let m3: m1 * m2; // compile-time multiplication checks!
```

As you can see in the example above, cayley's matrices are *generic*. The crate was designed with as much type flexibility in mind.
To give an example, for two matrices of different types `T` and `Q` to be multiplyable, there needs to exist a 'tensor type' `R` so that
an implementation of `Mul<Q, Output = R>` exists on T and an implementation of `Add` exists on `R`. For sensible numerical types these
implementations are automatically handled by the `num_traits` crate (which, incidentally, handles a lot of the mathematical logic in cayley).

## Contributions

I'm but a humble seventeen-year-old high schooler from Belgium who likes to write code in their free time. If you feel like helping out,
feel free to do so!
