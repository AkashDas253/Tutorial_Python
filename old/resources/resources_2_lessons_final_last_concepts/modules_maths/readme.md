
## Statistics

### Statistics module:

- The `statistics` module was new in Python 3.4 

### Statistics Methods

- `statistics.harmonic_mean(data)` - Calculates the harmonic mean (central location) of the given data. `data` is a sequence or iterable of numbers.
- `statistics.mean(data)` - Calculates the mean (average) of the given data. `data` is a sequence or iterable of numbers.
- `statistics.median(data)` - Calculates the median (middle value) of the given data. `data` is a sequence or iterable of numbers.
- `statistics.median_grouped(data, interval=1)` - Calculates the median of grouped continuous data. `data` is a sequence or iterable of numbers, and `interval` is the class interval (default is 1).
- `statistics.median_high(data)` - Calculates the high median of the given data. `data` is a sequence or iterable of numbers.
- `statistics.median_low(data)` - Calculates the low median of the given data. `data` is a sequence or iterable of numbers.
- `statistics.mode(data)` - Calculates the mode (central tendency) of the given numeric or nominal data. `data` is a sequence or iterable of numbers or strings.
- `statistics.pstdev(data, mu=None)` - Calculates the standard deviation from an entire population. `data` is a sequence or iterable of numbers, and `mu` is the mean of the population (optional).
- `statistics.stdev(data, xbar=None)` - Calculates the standard deviation from a sample of data. `data` is a sequence or iterable of numbers, and `xbar` is the mean of the sample (optional).
- `statistics.pvariance(data, mu=None)` - Calculates the variance of an entire population. `data` is a sequence or iterable of numbers, and `mu` is the mean of the population (optional).
- `statistics.variance(data, xbar=None)` - Calculates the variance from a sample of data. `data` is a sequence or iterable of numbers, and `xbar` is the mean of the sample (optional).

## Math Module

- Python has a built-in module that you can use for mathematical tasks.

- The math module has a set of methods and constants.

### Math Methods:

- `math.acos(x)` - Returns the arc cosine of a number.
- `math.acosh(x)` - Returns the inverse hyperbolic cosine of a number.
- `math.asin(x)` - Returns the arc sine of a number.
- `math.asinh(x)` - Returns the inverse hyperbolic sine of a number.
- `math.atan(x)` - Returns the arc tangent of a number in radians.
- `math.atan2(y, x)` - Returns the arc tangent of y/x in radians.
- `math.atanh(x)` - Returns the inverse hyperbolic tangent of a number.
- `math.ceil(x)` - Rounds a number up to the nearest integer.
- `math.comb(n, k)` - Returns the number of ways to choose k items from n items without repetition and order.
- `math.copysign(x, y)` - Returns a float consisting of the value of the first parameter and the sign of the second parameter.
- `math.cos(x)` - Returns the cosine of a number.
- `math.cosh(x)` - Returns the hyperbolic cosine of a number.
- `math.degrees(x)` - Converts an angle from radians to degrees.
- `math.dist(p, q)` - Returns the Euclidean distance between two points (p and q), where p and q are the coordinates of that point.
- `math.erf(x)` - Returns the error function of a number.
- `math.erfc(x)` - Returns the complementary error function of a number.
- `math.exp(x)` - Returns E raised to the power of x.
- `math.expm1(x)` - Returns Ex - 1.
- `math.fabs(x)` - Returns the absolute value of a number.
- `math.factorial(x)` - Returns the factorial of a number.
- `math.floor(x)` - Rounds a number down to the nearest integer.
- `math.fmod(x, y)` - Returns the remainder of x/y.
- `math.frexp(x)` - Returns the mantissa and the exponent, of a specified number.
- `math.fsum(iterable)` - Returns the sum of all items in any iterable (tuples, arrays, lists, etc.).
- `math.gamma(x)` - Returns the gamma function at x.
- `math.gcd(a, b)` - Returns the greatest common divisor of two integers.
- `math.hypot(*coordinates)` - Returns the Euclidean norm.
- `math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)` - Checks whether two values are close to each other, or not.
- `math.isfinite(x)` - Checks whether a number is finite or not.
- `math.isinf(x)` - Checks whether a number is infinite or not.
- `math.isnan(x)` - Checks whether a value is NaN (not a number) or not.
- `math.isqrt(n)` - Rounds a square root number downwards to the nearest integer.
- `math.ldexp(x, i)` - Returns the inverse of math.frexp() which is x * (2**i) of the given numbers x and i.
- `math.lgamma(x)` - Returns the log gamma value of x.
- `math.log(x, [base])` - Returns the natural logarithm of a number, or the logarithm of number to base.
- `math.log10(x)` - Returns the base-10 logarithm of x.
- `math.log1p(x)` - Returns the natural logarithm of 1+x.
- `math.log2(x)` - Returns the base-2 logarithm of x.
- `math.perm(n, k=None)` - Returns the number of ways to choose k items from n items with order and without repetition.
- `math.pow(x, y)` - Returns the value of x to the power of y.
- `math.prod(iterable, *, start=1)` - Returns the product of all the elements in an iterable.
- `math.radians(x)` - Converts a degree value into radians.
- `math.remainder(x, y)` - Returns the closest value that can make numerator completely divisible by the denominator.
- `math.sin(x)` - Returns the sine of a number.
- `math.sinh(x)` - Returns the hyperbolic sine of a number.
- `math.sqrt(x)` - Returns the square root of a number.
- `math.tan(x)` - Returns the tangent of a number.
- `math.tanh(x)` - Returns the hyperbolic tangent of a number.
- `math.trunc(x)` - Returns the truncated integer parts of a number.

### Math Constants:

- `math.e` - Returns Euler's number (2.7182...).
- `math.inf` - Returns a floating-point positive infinity.
- `math.nan` - Returns a floating-point NaN (Not a Number) value.
- `math.pi` - Returns PI (3.1415...).
- `math.tau` - Returns tau (6.2831...).

## cMath Methods

- Python has a built-in module that you can use for mathematical tasks for complex numbers.

- The methods in this module accepts int, float, and complex numbers. It even accepts Python objects that has a __complex__() or __float__() method.

- The methods in this module almost always return a complex number. If the return value can be expressed as a real number, the return value has an imaginary part of 0.

- The cmath module has a set of methods and constants.

### cMath Methods:

| Method            | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `cmath.acos(x)`   | Returns the arc cosine value of x                                           |
| `cmath.acosh(x)`  | Returns the hyperbolic arc cosine of x                                      |
| `cmath.asin(x)`   | Returns the arc sine of x                                                   |
| `cmath.asinh(x)`  | Returns the hyperbolic arc sine of x                                        |
| `cmath.atan(x)`   | Returns the arc tangent value of x                                          |
| `cmath.atanh(x)`  | Returns the hyperbolic arctangent value of x                                |
| `cmath.cos(x)`    | Returns the cosine of x                                                     |
| `cmath.cosh(x)`   | Returns the hyperbolic cosine of x                                          |
| `cmath.exp(x)`    | Returns the value of E^x, where E is Euler's number (approximately 2.718281...), and x is the number passed to it |
| `cmath.isclose()` | Checks whether two values are close, or not                                 |
| `cmath.isfinite(x)`| Checks whether x is a finite number                                        |
| `cmath.isinf(x)`  | Check whether x is a positive or negative infinity                          |
| `cmath.isnan(x)`  | Checks whether x is NaN (not a number)                                      |
| `cmath.log(x[, base])`| Returns the logarithm of x to the base                                  |
| `cmath.log10(x)`  | Returns the base-10 logarithm of x                                          |
| `cmath.phase()`   | Return the phase of a complex number                                        |
| `cmath.polar()`   | Convert a complex number to polar coordinates                               |
| `cmath.rect()`    | Convert polar coordinates to rectangular form                               |
| `cmath.sin(x)`    | Returns the sine of x                                                       |
| `cmath.sinh(x)`   | Returns the hyperbolic sine of x                                            |
| `cmath.sqrt(x)`   | Returns the square root of x                                                |
| `cmath.tan(x)`    | Returns the tangent of x                                                    |
| `cmath.tanh(x)`   | Returns the hyperbolic tangent of x                                         |

### cMath Constants

| Constant       | Description                                      |
|----------------|--------------------------------------------------|
| `cmath.e`      | Returns Euler's number (2.7182...)               |
| `cmath.inf`    | Returns a floating-point positive infinity value |
| `cmath.infj`   | Returns a complex infinity value                 |
| `cmath.nan`    | Returns floating-point NaN (Not a Number) value  |
| `cmath.nanj`   | Returns complex NaN (Not a Number) value         |
| `cmath.pi`     | Returns PI (3.1415...)                           |
| `cmath.tau`    | Returns tau (6.2831...)                          |

