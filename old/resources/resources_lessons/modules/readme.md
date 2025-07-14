
# Modules

## Python Modules

### What is a Module?

- A module is a code library.
- It is a file containing a set of functions and variables you want to include in your application.

### Create a Module

- Save the code you want in a file with the `.py` extension.

#### Syntax
```python
# Save this code in a file named mymodule.py
def greeting(name):
    print("Hello, " + name)
```

### Use a Module

- Use the `import` statement to use the module you created.

#### Syntax
```python
import mymodule

mymodule.greeting("Jonathan")
```

- Note: When using a function from a module, use the syntax: `module_name.function_name`.

### Variables in Module

- A module can contain functions and variables of all types (arrays, dictionaries, objects, etc.).

#### Syntax
```python
# Save this code in the file mymodule.py
person1 = {
    "name": "John",
    "age": 36,
    "country": "Norway"
}
```

#### Syntax
```python
import mymodule

a = mymodule.person1["age"]
print(a)
```

### Naming a Module

- You can name the module file whatever you like, but it must have the `.py` extension.

### Re-naming a Module

- Create an alias when you import a module using the `as` keyword.

#### Syntax
```python
import mymodule as mx

a = mx.person1["age"]
print(a)
```

### Built-in Modules

- Python has several built-in modules that you can import whenever you like.

#### Syntax
```python
import platform

x = platform.system()
print(x)
```

### Using the `dir()` Function

- The `dir()` function lists all the function names (or variable names) in a module.

#### Syntax
```python
import platform

x = dir(platform)
print(x)
```

- Note: The `dir()` function can be used on all modules, including the ones you create yourself.

### Import From Module

- Import only parts of a module using the `from` keyword.

#### Syntax
```python
# The module named mymodule has one function and one dictionary
def greeting(name):
    print("Hello, " + name)

person1 = {
    "name": "John",
    "age": 36,
    "country": "Norway"
}
```

#### Syntax
```python
from mymodule import person1

print(person1["age"])
```

- Note: When importing using the `from` keyword, do not use the module name when referring to elements in the module. Example: `person1["age"]`, not `mymodule.person1["age"]`.

## Python Datetime

### Python Dates

- A date in Python is not a data type of its own.
- We can import a module named `datetime` to work with dates as date objects.

### Importing the datetime Module

- Import the `datetime` module and display the current date:

#### Syntax
```python
import datetime

x = datetime.datetime.now()
print(x)
```

### Date Output

- The date contains year, month, day, hour, minute, second, and microsecond.
- Example output: `2024-10-27 14:36:17.450635`

### Methods in datetime Module

- The `datetime` module has many methods to return information about the date object.

#### Example: Return the year and name of weekday
```python
import datetime

x = datetime.datetime.now()

print(x.year)
print(x.strftime("%A"))
```

### Creating Date Objects

- To create a date, use the `datetime()` class (constructor) of the `datetime` module.
- The `datetime()` class requires three parameters to create a date: year, month, day.

#### Syntax
```python
import datetime

x = datetime.datetime(2020, 5, 17)

print(x)
```

- The `datetime()` class also takes parameters for time and timezone (hour, minute, second, microsecond, tzone), but they are optional and have a default value of 0 (None for timezone).

### The strftime() Method

- The `datetime` object has a method for formatting date objects into readable strings.
- The method is called `strftime()`, and takes one parameter, `format`, to specify the format of the returned string.

#### Example: Display the name of the month
```python
import datetime

x = datetime.datetime(2018, 6, 1)

print(x.strftime("%B"))
```

### Reference of All Legal Format Codes

| Directive | Description                                | Example       |
|-----------|--------------------------------------------|---------------|
| %a        | Weekday, short version                     | Wed           |
| %A        | Weekday, full version                      | Wednesday     |
| %w        | Weekday as a number 0-6, 0 is Sunday       | 3             |
| %d        | Day of month 01-31                         | 31            |
| %b        | Month name, short version                  | Dec           |
| %B        | Month name, full version                   | December      |
| %m        | Month as a number 01-12                    | 12            |
| %y        | Year, short version, without century       | 18            |
| %Y        | Year, full version                         | 2018          |
| %H        | Hour 00-23                                 | 17            |
| %I        | Hour 00-12                                 | 05            |
| %p        | AM/PM                                      | PM            |
| %M        | Minute 00-59                               | 41            |
| %S        | Second 00-59                               | 08            |
| %f        | Microsecond 000000-999999                  | 548513        |
| %z        | UTC offset                                 | +0100         |
| %Z        | Timezone                                   | CST           |
| %j        | Day number of year 001-366                 | 365           |
| %U        | Week number of year, Sunday as the first day of week, 00-53 | 52 |
| %W        | Week number of year, Monday as the first day of week, 00-53 | 52 |
| %c        | Local version of date and time             | Mon Dec 31 17:41:00 2018 |
| %C        | Century                                    | 20            |
| %x        | Local version of date                      | 12/31/18      |
| %X        | Local version of time                      | 17:41:00      |
| %%        | A % character                              | %             |
| %G        | ISO 8601 year                              | 2018          |
| %u        | ISO 8601 weekday (1-7)                     | 1             |
| %V        | ISO 8601 weeknumber (01-53)                | 01            |

## Python Math

### Built-in Math Functions

- Python has a set of built-in math functions and an extensive math module for performing mathematical tasks on numbers.

#### `min()` and `max()`

- These functions find the lowest or highest value in an iterable.

#### Syntax
```python
x = min(value1, value2, ...)
y = max(value1, value2, ...)
print(x)
print(y)
```

#### `abs()`

- This function returns the absolute (positive) value of the specified number.

#### Syntax
```python
x = abs(number)
print(x)
```

#### `pow(x, y)`

- This function returns the value of `x` to the power of `y` (x^y).

#### Syntax
```python
x = pow(base, exponent)
print(x)
```

### The Math Module

- Python has a built-in module called `math` that extends the list of mathematical functions.
- To use it, you must import the `math` module.

#### Importing the Math Module
```python
import math
```

#### `math.sqrt()`

- This method returns the square root of a number.

#### Syntax
```python
import math
x = math.sqrt(number)
print(x)
```

#### `math.ceil()` and `math.floor()`

- `math.ceil()` rounds a number upwards to its nearest integer.
- `math.floor()` rounds a number downwards to its nearest integer.

#### Syntax
```python
import math
x = math.ceil(number)
y = math.floor(number)
print(x)  # rounded up
print(y)  # rounded down
```

#### `math.pi`

- This constant returns the value of PI (3.14...).

#### Syntax
```python
import math
x = math.pi
print(x)
```



## Random Module 

The `random` module in Python provides a suite of methods for generating random numbers and performing random operations.

### Random Method

- `random.seed(a=None, version=2)` - Initialize the random number generator. `a` is the seed value, and `version` specifies the version of the seeding algorithm.
- `random.getstate()` - Returns the current internal state of the random number generator.
- `random.setstate(state)` - Restores the internal state of the random number generator. `state` is the state object to restore.
- `random.getrandbits(k)` - Returns a number representing `k` random bits.
- `random.randrange(start, stop=None, step=1)` - Returns a random number between the given range. `start` is the starting value, `stop` is the ending value (exclusive), and `step` is the increment.
- `random.randint(a, b)` - Returns a random integer between `a` and `b` (both inclusive).
- `random.choice(seq)` - Returns a random element from the given sequence. `seq` is the sequence to choose from.
- `random.choices(population, weights=None, *, cum_weights=None, k=1)` - Returns a list with a random selection from the given sequence. `population` is the sequence to choose from, `weights` is a list of relative weights, `cum_weights` is a list of cumulative weights, and `k` is the number of elements to choose.
- `random.shuffle(x, random=None)` - Takes a sequence and returns the sequence in a random order. `x` is the sequence to shuffle, and `random` is an optional random function.
- `random.sample(population, k)` - Returns a given sample of a sequence. `population` is the sequence to sample from, and `k` is the number of elements to sample.
- `random.random()` - Returns a random float number between 0 and 1.
- `random.uniform(a, b)` - Returns a random float number between `a` and `b`.
- `random.triangular(low, high, mode)` - Returns a random float number between `low` and `high`, with `mode` specifying the midpoint.
- `random.betavariate(alpha, beta)` - Returns a random float number between 0 and 1 based on the Beta distribution. `alpha` and `beta` are the shape parameters.
- `random.expovariate(lambd)` - Returns a random float number based on the Exponential distribution. `lambd` is the rate parameter.
- `random.gammavariate(alpha, beta)` - Returns a random float number based on the Gamma distribution. `alpha` is the shape parameter, and `beta` is the scale parameter.
- `random.gauss(mu, sigma)` - Returns a random float number based on the Gaussian distribution. `mu` is the mean, and `sigma` is the standard deviation.
- `random.lognormvariate(mu, sigma)` - Returns a random float number based on a log-normal distribution. `mu` is the mean of the underlying normal distribution, and `sigma` is the standard deviation.
- `random.normalvariate(mu, sigma)` - Returns a random float number based on the normal distribution. `mu` is the mean, and `sigma` is the standard deviation.
- `random.vonmisesvariate(mu, kappa)` - Returns a random float number based on the von Mises distribution. `mu` is the mean direction, and `kappa` is the concentration parameter.
- `random.paretovariate(alpha)` - Returns a random float number based on the Pareto distribution. `alpha` is the shape parameter.
- `random.weibullvariate(alpha, beta)` - Returns a random float number based on the Weibull distribution. `alpha` is the scale parameter, and `beta` is the shape parameter.
