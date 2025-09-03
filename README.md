[![Python Template for IDS706](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/main.yml/badge.svg)](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/main.yml)

# JDS706_DE_Wk1.2

## Python Template Overview
This is the first homework for **IDS 706: Data Engineering Systems**.  
It includes:

- A Python script with simple functions (`hello.py`)
- Unit tests with `pytest` (`test_hello.py`)
- Code formatting with `black` and linting with `flake8`
- A `Makefile` for setup, testing, formatting, and cleanup
- CI with GitHub Actions

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/josephong610/JDS706_DE_Wk1.2.git
   ```

2. **Setup Python Environment**
   ```bash
   python3 -m venv ~/.IDS706_python_template
   source ~/.IDS706_python_template/bin/activate
   ```

3. **Makefile commands**
   ```bash
   make install    # install dependencies
   make format     # format code
   make lint       # lint code
   make test       # run tests
   make clean      # remove temporary files
   ```

---

## Usage

1. **Run the main script**
   ```bash
   python3 hello.py   # This has a function for welcoming someone using their name and another function that simply multiplies two numbers together
   ```
   Expected output:
   ```
   Hello, Prof. Yu, welcome to Data Engineering Systems (IDS 706)!
   2 * 3 = 6
   ```

2. **Use functions in Python**
   ```python
   from hello import say_hello, multiply

   print(say_hello("Joseph"))
   print(multiply(5, 4))
   ```

3. **Run tests**
   ```bash
   make test
   ```

---

## Project Structure
```
hello.py        # Main script with functions
test_hello.py   # Unit tests
requirements.txt
Makefile
README.md
```

---

This project includes **automated CI/CD**, so linting and tests run automatically on every push via GitHub Actions.
