# Contributing to `ensemble_md`

Thank you for your interest in contributing to `ensemble_md`! We welcome contributions from the community and appreciate your efforts to improve the project. Please follow these guidelines to help us manage contributions effectively.

## General workflow
We use a "Pull request, Review, and Merge" workflow for contributions to this project: codes can only be added to the master branch through pull requests (PRs). To contribute to the project, please follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top right of this repository to create your own copy of the project.
2. **Clone Your Fork**: Clone your fork to your local machine:
   ```bash
   git clone https://github.com/{your-username}/ensemble_md.git
   ```
3. ***Create a Branch***: Create a new branch for your changes:
    ```bash
    git checkout -b {your-feature-branch}
    ```
4. **Make Changes**: Make your changes to the project, and commit them to your branch using Git:
    ```bash
    git add .
    git commit -m "Your commit message"
    ```
5. **Push Changes**: Push your changes to your fork on GitHub:
    ```bash
    git push origin {your-feature-branch}
    ```
6. **Create a Pull Request**: Go to the GitHub page of your fork and create a new pull request. Make sure to provide a clear description of your changes in the PR. Check the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) for more information on what to include in your PR.

## Contribution Guidelines

### Coding Style
We use [PEP8 coding style](https://peps.python.org/pep-0008/) for this project. Please follow the guidelines to maintain consistency in code style. Before you submit a pull request, run the following command in the root directory of the project to check your code for PEP8 compliance:
  ```bash
  flake8 ensemble_md
  ``` 
  Any errors or warnings should be fixed before submitting your pull request. (We accept PRs with `E501` and `E741` errors, but the errors should be suppressed by appending `# noqa: E501` or `#noqa: E741` at the end of the line.)

### Docstrings and documentation
Whenever there is a new function or class added, please include a docstring that describes the purpose of the function or class. For this project, we follow the [NumPy docstrings standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). Please visit the docstrings of existing functions or classes for reference.

If your changes affect any part of the project's functionality or usage, or add new features or algorithms to the project, please update the documentation in the `docs` directory accordingly. We use [Sphinx](https://www.sphinx-doc.org/en/master/) to generate documentation. Please ensure that your changes do not break the documentation build. You can build the documentation locally by running the following command in the `docs` directory:
```bash
make html
```
and then opening the `docs/build/html/index.html` file in your browser to view the documentation. If you implement a new simulation method in `ensemble_md`, we encourage you to include a tutorial in the documentation by including a `.ipynb` file in the `docs/example` folder. Ideally, the tutorial should be able to be run on [Binder](https://mybinder.org/), for which additional dependencies (if any) should be added to `environment.yml` in the root directory of the project.

### Unit tests and continuous integration
To maintain the quality of the project, we require that the merging code be covered by unit tests to at least 90% coverage. We use [pytest](https://docs.pytest.org/en/stable/) for unit testing. Please add new tests for new features or changes to existing features. You can run the tests locally by running the following command in the root directory of the project:
```bash
pytest -vv --disable-pytest-warnings --cov=ensemble_md --cov-report=xml --color=yes ensemble_md/tests/
```
This will generate a coverage report in the `coverage.xml` file in the root directory of the project. To view the coverage report, you can run the following command:
```bash
coverage report
```
For tests that require the use of MPI, you might want to use a command like the following to run the tests:
```bash
mpirun -np 4 pytest -vv --disable-pytest-warnings --cov=ensemble_md --cov-report=xml --color=yes ensemble_md/tests/{tests_that_use_mpi.py} --with-mpi
```
For continuous integration (CI), we use [CircleCI](https://circleci.com/) to run tests on every pull request and push to the repository. Make any necessary changes to the CI configuration file (`.circleci/config.yml`) to ensure that your changes pass the CI tests before submitting a pull request.

## Reporting issues
If you encounter any issues with the project, please report them by opening an issue on the GitHub repository. When reporting an issue, please provide a clear description of the problem, including the steps to reproduce the issue, the expected behavior, and the actual behavior. If possible, include any error messages or stack traces that you encountered. Please refer to the [issue template](.github/ISSUE_TEMPLATE.md) for more information on what to include in your issue report.

## Code of Conduct
By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment for everyone.

## Contact
For any questions or further assistance, please don't hesitate to contact us as wehs7661@colorado.edu.

---
Thank you for your contributions to `ensemble_md`! We appreciate your help in making this project better for everyone.

**`ensemble_md` Development Team**
